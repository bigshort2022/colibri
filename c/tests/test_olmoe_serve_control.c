/* Production reader/queue transitions, without model weights. */
#include <stdio.h>
#include <stdarg.h>
static int boundary_printf(const char *format, ...);
static int boundary_fprintf(FILE *stream, const char *format, ...);
#define printf boundary_printf
#define fprintf boundary_fprintf
#define OLMOE_TESTING 1
#define main coli_olmoe_main_unused
#include "../olmoe.c"
#undef main
#undef printf
#undef fprintf
#include <assert.h>

static int failures;
#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); failures++; \
} } while (0)

static FILE *input_bytes(const char *bytes) {
    FILE *input = tmpfile(); assert(input);
    assert(fwrite(bytes, 1, strlen(bytes), input) == strlen(bytes));
    rewind(input); return input;
}
static int command(const char *bytes, FILE *output, const char *active) {
    FILE *input = input_bytes(bytes);
    int result = serve_read_cmd(input, output, active);
    fclose(input); return result;
}
static void clear_queue(void) {
    while (g_qn) free(g_q[--g_qn].payload);
    memset(g_q, 0, sizeof(g_q));
}
static void expect_output(FILE *output, const char *expected) {
    char bytes[2048]; fflush(output); rewind(output);
    size_t n = fread(bytes, 1, sizeof(bytes)-1, output); bytes[n] = 0;
    CHECK(strcmp(bytes, expected) == 0);
}
static void test_active_controls(void) {
    FILE *output = tmpfile(); assert(output);
    CHECK(command("STOP live\n", output, "live") == 1);
    CHECK(command("CANCEL live\n", output, "live") == 3);
    CHECK(command("CANCEL old\n", output, "live") == 0);
    CHECK(command("STOP old\n", output, "live") == 0);
    expect_output(output, "ERROR old NOT_FOUND\nERROR old NOT_FOUND\n");
    fclose(output);
}
static void test_fifo_controls(void) {
    FILE *output = tmpfile(); assert(output);
    CHECK(command("SUBMIT first 0 3 4 0 1 logprobs=2 pin=1\na\nb\n", output, "live") == 0);
    CHECK(command("SUBMIT middle 0 1 4 0 1\nx\n", output, "live") == 0);
    CHECK(command("SUBMIT last 0 1 4 0 1\nz\n", output, "live") == 0);
    CHECK(g_qn == 3);
    CHECK(command("STOP middle\n", output, "live") == 0);
    if (g_qn == 3) CHECK(g_q[1].stop_requested == 1 && g_q[0].stop_requested == 0);
    CHECK(command("CANCEL middle\n", output, "live") == 0);
    CHECK(g_qn == 2);
    if (g_qn == 2) {
        CHECK(strcmp(g_q[0].id, "first") == 0 && strcmp(g_q[1].id, "last") == 0);
        CHECK(g_q[0].logprobs == 2 && g_q[0].pin == 1);
        CHECK(memcmp(g_q[0].payload, "a\nb", 3) == 0);
    }
    expect_output(output, "ERROR middle CANCELLED\n");
    clear_queue(); fclose(output);
}
static void test_input_outcomes(void) {
    FILE *output = tmpfile(); assert(output);
    CHECK(command("", output, "live") == -1);
    CHECK(command("SUBMIT broken 0 1 4 0 1\nxX", output, "live") == -2);
    CHECK(command("CANCEL live", output, "live") == -2);
    CHECK(command("CANCEL live garbage\n", output, "live") == -2);
    CHECK(command("STOP live garbage\n", output, "live") == -2);
    CHECK(g_qn == 0);
    fclose(output);
}
#ifdef _WIN32
#define test_dup _dup
#define test_dup2 _dup2
#define test_close _close
#define test_write _write
#define test_pipe(fd) _pipe(fd, 4096, _O_BINARY)
#else
#define test_dup dup
#define test_dup2 dup2
#define test_close close
#define test_write write
#define test_pipe(fd) pipe(fd)
#endif

static int boundary_fd = -1;
static const char *boundary_format, *boundary_frame;
static void inject_boundary(const char *format) {
    if (boundary_fd >= 0 && boundary_frame && !strcmp(format, boundary_format)) {
        size_t n = strlen(boundary_frame);
        CHECK(test_write(boundary_fd, boundary_frame, (unsigned)n) == (int)n);
        boundary_frame = NULL;
    }
}
static int boundary_printf(const char *format, ...) {
    va_list ap; va_start(ap, format); int n = vprintf(format, ap); va_end(ap);
    inject_boundary(format); return n;
}
static int boundary_fprintf(FILE *stream, const char *format, ...) {
    va_list ap; va_start(ap, format); int n = vfprintf(stream, format, ap); va_end(ap);
    if (stream == stdout) inject_boundary(format);
    return n;
}
static void set_input(const char *bytes) {
    FILE *input = input_bytes(bytes);
    assert(test_dup2(fileno(input), fileno(stdin)) >= 0);
    fclose(input); clearerr(stdin); g_input_closed = 0;
}
static void set_boundary(const char *format, const char *frame) {
    int fd[2]; assert(test_pipe(fd) == 0);
    assert(test_dup2(fd[0], fileno(stdin)) >= 0);
    test_close(fd[0]); clearerr(stdin); g_input_closed = 0;
    boundary_fd = fd[1]; boundary_format = format; boundary_frame = frame;
}
static FILE *capture_output(void) {
    fflush(stdout); FILE *output = tmpfile(); assert(output);
    assert(test_dup2(fileno(output), fileno(stdout)) >= 0);
    return output;
}
static void output_text(FILE *output, char *bytes, size_t capacity) {
    fflush(stdout); rewind(output);
    size_t n = fread(bytes, 1, capacity-1, output); bytes[n] = 0;
}
static void test_transaction_boundaries(void) {
    int old_in = test_dup(fileno(stdin)), old_out = test_dup(fileno(stdout));
    assert(old_in >= 0 && old_out >= 0);
    char bytes[8192]; float one = 1; char *names[] = {"x"}; int added[] = {1};
    int special_flags[] = {0};
    Special special = {.str="x", .len=1, .id=0};
    Tok tokenizer = {.sp=&special, .nsp=1, .id2str=names, .id_added=added,
                     .id_special=special_flags, .n_ids=1};
    Model model = {0}; model.c.hidden=1; model.c.vocab=1; model.c.eps=1e-6f;
    model.embed=&one; model.final_norm=&one; model.lm_head=&one;
    assert(kv_prefix_alloc(&model.kvp, 16));
    SReq request = {.id="live", .payload="x", .plen=1, .max_tok=2};
    FILE *output = capture_output(); set_input("");
    CHECK(serve_one(&model, &tokenizer, &request, 16) == 0);
    output_text(output, bytes, sizeof(bytes));
    CHECK(strstr(bytes, "ACCEPT live 1\n") && strstr(bytes, "DONE live STAT 2 "));
    CHECK(g_input_closed && model.kv_len == 2 && model.kvp.len == 2 && model.token_count == 2);
    fclose(output);

    output = capture_output(); set_boundary("ACCEPT %s %d\n", "CANCEL live\n");
    int computed = model.token_count;
    CHECK(serve_one(&model, &tokenizer, &request, 16) == 0);
    test_close(boundary_fd); boundary_fd=-1; output_text(output, bytes, sizeof(bytes));
    CHECK(model.token_count == computed && !strstr(bytes, "DATA "));
    CHECK(strstr(bytes, "ERROR live CANCELLED\n") && !strstr(bytes, "DONE "));
    fclose(output);

    const char *controls[] = {"STOP live\nCANCEL live\n", "STOP live\n"};
    for (int i=0; i<2; i++) {
        output = capture_output(); set_boundary("DATA %s %zu\n", controls[i]);
        CHECK(serve_one(&model, &tokenizer, &request, 16) == 0);
        test_close(boundary_fd); boundary_fd=-1; output_text(output, bytes, sizeof(bytes));
        char *data = strstr(bytes, "DATA live ");
        CHECK(data && !strstr(data+1, "DATA live "));
        CHECK(model.kv_len == 1 && model.kvp.len == 1);
        if (!i) CHECK(strstr(bytes, "ERROR live CANCELLED\n") && !strstr(bytes, "DONE "));
        else CHECK(strstr(bytes, "DONE live STAT 1 ") && strstr(bytes, " 1 0\n") && !strstr(bytes, "ERROR "));
        fclose(output);
    }

    output = capture_output(); set_boundary("ECHO %s %d %d%s\n", "CANCEL live\n");
    request.payload="xx"; request.plen=2; request.logprobs=1;
    kv_prefix_clear(&model.kvp);
    CHECK(serve_one(&model, &tokenizer, &request, 16) == 0);
    test_close(boundary_fd); boundary_fd=-1; output_text(output, bytes, sizeof(bytes));
    CHECK(strstr(bytes, "ECHO live ") && !strstr(bytes, "DATA "));
    CHECK(strstr(bytes, "ERROR live CANCELLED\n") && !strstr(bytes, "DONE "));
    CHECK(model.kvp.len == 2);
    fclose(output); request.payload="x"; request.plen=1; request.logprobs=0;

    output = capture_output(); set_input("CANCEL old\n");
    snprintf(request.id, sizeof(request.id), "next");
    CHECK(serve_one(&model, &tokenizer, &request, 16) == 0);
    output_text(output, bytes, sizeof(bytes));
    CHECK(strstr(bytes, "ERROR old NOT_FOUND\n") && strstr(bytes, "DONE next STAT 2 "));
    fclose(output);
    output = capture_output(); set_input("");
    CHECK(serve_one(&model, &tokenizer, &request, 2) == 0);
    output_text(output, bytes, sizeof(bytes));
    CHECK(strstr(bytes, "ERROR next CONTEXT_EXCEEDED 3 2\n") && !strstr(bytes, "ACCEPT "));
    fclose(output);

    output = capture_output(); set_input("");
    CHECK(command("SUBMIT first 0 1 2 0 1\nx\n", stdout, NULL) == 0);
    CHECK(command("SUBMIT last 0 1 2 0 1\nx\n", stdout, NULL) == 0);
    CHECK(command("STOP first\n", stdout, NULL) == 0);
    CHECK(command("STOP last\n", stdout, NULL) == 0);
    computed = model.token_count;
    serve_loop(&model, &tokenizer, 16);
    output_text(output, bytes, sizeof(bytes));
    CHECK(g_qn == 0 && g_input_closed && model.token_count == computed);
    CHECK(strstr(bytes, "ACCEPT first 1\n") && strstr(bytes, "DONE first STAT 0 "));
    CHECK(strstr(bytes, "ACCEPT last 1\n") && strstr(bytes, "DONE last STAT 0 "));
    CHECK(!strstr(bytes, "DATA "));
    fclose(output);

    output = capture_output();
    set_input("SUBMIT queued 0 1 2 0 1\nx\nCANCEL first garbage\n");
    CHECK(command("SUBMIT first 0 1 2 0 1\nx\n", stdout, NULL) == 0);
    serve_loop(&model, &tokenizer, 16);
    output_text(output, bytes, sizeof(bytes));
    CHECK(g_qn == 0 && g_input_closed && !strstr(bytes, "DONE "));
    CHECK(strstr(bytes, "ERROR first BAD_FRAME\n") && !strstr(bytes, "ERROR queued"));
    fclose(output);
    kv_prefix_free(&model.kvp); free(model.ehit);
    assert(test_dup2(old_in, fileno(stdin)) >= 0);
    assert(test_dup2(old_out, fileno(stdout)) >= 0);
    test_close(old_in); test_close(old_out); clearerr(stdin);
}

/* Exercise real stdin polling and serve_one's no-forward stopped path. */
static void test_poll_and_stopped_admission(void) {
    const char *controls[] = {
        "STOP live\nCANCEL live\n", "CANCEL live\nSTOP live\n",
        "STOP live\nSUBMIT live 0 1 4 0 1\nxX",
    };
    const int outcomes[] = {3, 3, -2};
    int old_in = test_dup(fileno(stdin)), old_out = test_dup(fileno(stdout));
    assert(old_in >= 0 && old_out >= 0);
    fflush(stdout);
    for (size_t i = 0; i < sizeof(outcomes)/sizeof(outcomes[0]); i++) {
        FILE *input = input_bytes(controls[i]), *output = tmpfile(); assert(output);
        assert(test_dup2(fileno(input), fileno(stdin)) >= 0);
        assert(test_dup2(fileno(output), fileno(stdout)) >= 0);
        clearerr(stdin); g_input_closed = 0;
        CHECK(serve_poll_controls("live") == outcomes[i]);
        CHECK(g_input_closed);
        CHECK(serve_poll_controls("live") == 0);
        ColiServeDone done = {.completion_tokens=2, .prompt_tokens=5, .length_limited=1};
        serve_emit_terminal(stdout, "live", outcomes[i], &done);
        expect_output(output, outcomes[i] == 3 ? "ERROR live CANCELLED\n" : "ERROR live BAD_FRAME\n");
        fclose(input); fclose(output);
    }
    FILE *input = input_bytes(""), *output = tmpfile(); assert(output);
    assert(test_dup2(fileno(input), fileno(stdin)) >= 0);
    assert(test_dup2(fileno(output), fileno(stdout)) >= 0);
    clearerr(stdin); g_input_closed = 0;
    Special special = {.str="x", .len=1, .id=0};
    Tok tokenizer = {.sp=&special, .nsp=1};
    Model model = {0};
    SReq request = {.id="stopped", .payload="x", .plen=1, .max_tok=4, .stop_requested=1};
    CHECK(serve_one(&model, &tokenizer, &request, 16) == 0);
    CHECK(g_input_closed && model.kvp.len == 0 && g_prof_forwards == 0);
    expect_output(output, "ACCEPT stopped 1\nDONE stopped STAT 0 0.000 0.0 0.00 1 0\n");
    fclose(input); fclose(output);
    assert(test_dup2(old_in, fileno(stdin)) >= 0);
    assert(test_dup2(old_out, fileno(stdout)) >= 0);
    test_close(old_in); test_close(old_out); clearerr(stdin);
}

static void test_terminal_outcomes(void) {
    const int controls[] = {0, 1, 3, -1, -2};
    const char *expected[] = {
        "DONE live STAT 2 3.000 4.0 5.00 6 1\n",
        "DONE live STAT 2 3.000 4.0 5.00 6 0\n",
        "ERROR live CANCELLED\n",
        "DONE live STAT 2 3.000 4.0 5.00 6 1\n",
        "ERROR live BAD_FRAME\n",
    };
    for (size_t i = 0; i < sizeof(controls)/sizeof(controls[0]); i++) {
        FILE *output = tmpfile(); assert(output);
        ColiServeDone done = {2, 3, 4, 5, 6, 1};
        serve_emit_terminal(output, "live", controls[i], &done);
        expect_output(output, expected[i]); fclose(output);
    }
}
typedef struct { const char *bytes; size_t count; int fd; } FragmentInput;
static void *write_fragmented(void *opaque) {
    FragmentInput *input = opaque;
    for (size_t i=0; i<input->count; i++)
        if (test_write(input->fd, input->bytes+i, 1) != 1) {
            test_close(input->fd); return (void *)1;
        }
    test_close(input->fd); return NULL;
}
static void test_fragmented_payload(void) {
    static const char frame[] = "SUBMIT unicode 0 7 2 0 1\n\xc3\xa9\n\xf0\x9f\x99\x82\n";
    int fd[2]; assert(test_pipe(fd) == 0);
    FILE *input = fdopen(fd[0], "rb"), *output = tmpfile(); assert(input && output);
    FragmentInput fragment = {frame, sizeof(frame)-1, fd[1]};
    pthread_t writer; assert(pthread_create(&writer, NULL, write_fragmented, &fragment) == 0);
    g_input_closed = 0;
    CHECK(serve_read_cmd(input, output, NULL) == 0 && g_qn == 1);
    if (g_qn == 1)
        CHECK(g_q[0].plen == 7 && memcmp(g_q[0].payload, "\xc3\xa9\n\xf0\x9f\x99\x82", 7) == 0);
    CHECK(serve_read_cmd(input, output, NULL) == -1 && g_qn == 1);
    void *result; assert(pthread_join(writer, &result) == 0); CHECK(result == NULL);
    expect_output(output, "");
    clear_queue(); fclose(input); fclose(output);
}
int main(void) {
    test_active_controls(); test_fifo_controls(); test_input_outcomes();
    setvbuf(stdin, NULL, _IONBF, 0);
    test_poll_and_stopped_admission(); test_terminal_outcomes();
    test_transaction_boundaries();
    test_fragmented_payload();
    if (failures) return 1;
    puts("olmoe serve control: ok"); return 0;
}
