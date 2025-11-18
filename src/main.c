#include "app.h"

#include "srsm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>

#include <sys/types.h>
#include <sys/time.h>

#include <malloc.h>
#include <time.h>
#include <unistd.h>



#define RUN_C_NO_MMAP_INCLUDE

#define RUN_C_NO_CONFIG
#define RUN_C_NO_TRANSFORMERWEIGHTS
#define RUN_C_NO_RUNSTATE
#define RUN_C_NO_TRANSFORMER

/*#define RUN_C_NO_MEMORY_MAP_WEIGHTS*/

#define RUN_C_NO_READ_CHECKPOINT
#define RUN_C_NO_FREE_TRANSFORMER

#define RUN_C_NO_RMSNORM
#define RUN_C_NO_SOFTMAX
#define RUN_C_NO_MATMUL

#define RUN_C_NO_TOKENINDEX
#define RUN_C_NO_TOKENIZER

#define RUN_C_NO_BUILD_TOKENIZER

#define RUN_C_NO_TIME_IN_MS

#define RUN_C_NO_READ_STDIN

#define TESTING


typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size);

void free_transformer(Transformer* t);

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size);

typedef void (*rmsnorm_t)(float*, float*, float*, int);
typedef void (*softmax_t)(float*, int);
typedef void (*matmul_t)(float*, float*, float*, int, int);

rmsnorm_t rmsnorm = NULL;
softmax_t softmax = NULL;
matmul_t matmul = NULL;

long time_in_ms();

void read_stdin(const char* guide, char* buffer, size_t bufsize);

#include "../trholding_llama2.c/run.c"



int main(int argc, char *argv[])
{

    printf("APPID %s, BASE_PATH %s\n", APPID, BASE_PATH);



    rmsnorm = scalar_rmsnorm;

    softmax = scalar_in_place_softmax;

    matmul = scalar_matmul;



    // default parameters
    char *checkpoint_path = BASE_PATH "stories15M.bin";  // e.g. out/model.bin
    char *tokenizer_path = BASE_PATH "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = "One day, Lily met a Shoggoth";        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode


    if (llamaver == 3){ rope_tf = 500000.0; }


    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);


    generate(&transformer, &tokenizer, &sampler, prompt, steps);


    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);

    return 0;

}


void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size)
{

    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }


    config->dim        = __builtin_bswap32(config->dim);
    config->hidden_dim = __builtin_bswap32(config->hidden_dim);
    config->n_layers   = __builtin_bswap32(config->n_layers);
    config->n_heads    = __builtin_bswap32(config->n_heads);
    config->n_kv_heads = __builtin_bswap32(config->n_kv_heads);
    config->vocab_size = __builtin_bswap32(config->vocab_size);
    config->seq_len    = __builtin_bswap32(config->seq_len);


    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes


    *data = malloc(*file_size);
    fseek(file, 0, SEEK_SET);
    fread(*data, *file_size, 1, file);


    fclose(file);


    *fd = -1;


    float *weights_ptr = *data + sizeof(Config) / sizeof(float);


    for (size_t i = 0; i < (*file_size - sizeof(Config)) / sizeof(float); ++i)
    {

        unsigned int *p = (unsigned int *)&weights_ptr[i];

        *p = __builtin_bswap32(*p);

    }


    memory_map_weights(weights, config, weights_ptr, shared_weights);

}

void free_transformer(Transformer* t) {

    if (t->data != NULL)
    {
        free(t->data);
    }
    free_run_state(&t->state);

}


void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }


    t->max_token_length = __builtin_bswap32(t->max_token_length);


    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }


        len = __builtin_bswap32(len);


        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}


/* TEMP: */
void read_stdin(const char* guide, char* buffer, size_t bufsize)
{
    (void)guide;
    (void)bufsize;
    strcpy(buffer, "One day, Lily met a Shoggoth");
}


long time_in_ms()
{

    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long)(tv.tv_sec * 1000L + tv.tv_usec / 1000L);

}
