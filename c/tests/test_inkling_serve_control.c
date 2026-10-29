/* Exercise the production serial protocol without model weights. */
#include <stdio.h>
#include <stdarg.h>
static int test_printf(const char *format, ...);
#define printf test_printf
#define main inkling_main_unused
#include "../inkling.c"
#undef main
#undef printf
#include <unistd.h>
#include <sys/wait.h>

static int failures;
#define CHECK(c, msg) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } } while (0)
static FILE *wire;
/* Inject controls at actual production wire boundaries, without a timing sleep. */
static int control_fd=-1;
static const char *control_event, *control_frame;
static int test_printf(const char *format, ...) {
    va_list ap; va_start(ap,format); int n=vprintf(format,ap); va_end(ap);
    if (control_fd>=0 && control_frame && !strcmp(format,control_event)) {
        size_t len=strlen(control_frame);
        CHECK(write(control_fd,control_frame,len)==(ssize_t)len,"inject control");
        control_frame=NULL;
    }
    return n;
}
static void boundary_input(const char *event, const char *frame) {
    int fd[2]; CHECK(pipe(fd)==0,"control pipe");
    dup2(fd[0],fileno(stdin)); close(fd[0]); clearerr(stdin);
    g_input_closed=g_input_bad=0; control_fd=fd[1]; control_event=event; control_frame=frame;
}
static void input(const void *bytes, size_t n) {
    FILE *f = tmpfile(); fwrite(bytes, 1, n, f); rewind(f);
    g_input_closed = g_input_bad = 0;
    dup2(fileno(f), fileno(stdin)); fclose(f); clearerr(stdin);
}
static void text_input(const char *s) { input(s, strlen(s)); }
static void capture(void) {
    fflush(stdout); wire = tmpfile(); dup2(fileno(wire), fileno(stdout)); clearerr(stdout);
}
static void output(char *out, size_t cap) {
    fflush(stdout); rewind(wire); size_t n = fread(out, 1, cap-1, wire); out[n] = 0;
}
static void clear_queue(void) {
    for (int i=0; i<g_qn; i++) { free(g_q[i].payload); free(g_q[i].audio); }
    g_qn=0;
}
static void controls(void) {
    char out[2048]; capture();
    serve_emit_terminal("cancel", 3, 2, 1, 0, 0, 4, 1);
    serve_emit_terminal("stop", 1, 2, 1, 0, 0, 4, 1);
    serve_emit_terminal("bad", -2, 2, 1, 0, 0, 4, 1);
    output(out,sizeof(out));
    CHECK(strstr(out,"ERROR cancel CANCELLED\n") != NULL,"CANCEL terminal");
    CHECK(strstr(out,"DONE cancel") == NULL,"CANCEL must not succeed");
    CHECK(strstr(out,"DONE stop STAT 2 1.000 0.0 0.00 4 0\n") != NULL,"STOP clears length limit");
    CHECK(strstr(out,"ERROR bad BAD_FRAME\n") != NULL,"fatal terminal");
    fclose(wire);
    capture(); text_input("STOP active\nCANCEL active\nCANCEL missing\n");
    CHECK(serve_read_cmd("active")==1,"active STOP");
    CHECK(serve_read_cmd("active")==3,"active CANCEL");
    CHECK(serve_read_cmd("active")==0,"late CANCEL");
    CHECK(serve_read_cmd("active")==-1,"EOF distinct");
    output(out,sizeof(out)); CHECK(strstr(out,"ERROR missing NOT_FOUND\n") != NULL,"late control error"); fclose(wire);
    capture(); text_input("SUBMIT a 0 1 2 0 1\nx\nSUBMIT b 0 1 2 0 1\ny\nSTOP a\nCANCEL a\n");
    CHECK(serve_read_cmd(NULL)==0 && serve_read_cmd(NULL)==0,"queue submissions");
    CHECK(g_qn==2,"two queued requests");
    CHECK(serve_read_cmd(NULL)==0,"queued STOP");
    CHECK(g_q[0].stop_requested == 1,"queued STOP flag");
    CHECK(serve_read_cmd(NULL)==0,"queued CANCEL");
    CHECK(g_qn==1 && !strcmp(g_q[0].id,"b"),"cancel preserves remaining FIFO");
    CHECK(serve_read_cmd(NULL)==-1 && g_qn==1,"EOF retains queued work");
    output(out,sizeof(out)); CHECK(strstr(out,"ERROR a CANCELLED\n") != NULL,"queued CANCEL terminal"); fclose(wire); clear_queue();
}
static void framing(void) {
    char out[2048]; capture();
    text_input("SUBMIT reject 0 14 -1 0 1\nCANCEL active\n\nSUBMIT next 0 1 2 0 1\nx\n");
    CHECK(serve_read_cmd("active")==0,"semantic reject consumes embedded control");
    CHECK(serve_read_cmd("active")==0 && g_qn==1 && !strcmp(g_q[0].id,"next"),"safe request after semantic reject");
    clear_queue();
    text_input("SUBMIT ext 0 1 2 0 1 14\nxCANCEL active\n\nSUBMIT next 0 1 2 0 1\nx\n");
    CHECK(serve_read_cmd("active")==0 && g_qn==1,"audio extension accepted");
    CHECK(g_qn && g_q[0].alen==14 && !memcmp(g_q[0].audio,"CANCEL active\n",14),"audio remains payload");
    CHECK(serve_read_cmd("active")==0 && g_qn==2,"safe request after extension");
    clear_queue();
    const char *bad[] = {
        "SUBMIT a 0 1 2 0 1\nx!", "SUBMIT a 0 3 2 0 1\nx",
        "SUBMIT a 0 -1 2 0 1\n", "SUBMIT a 0 999999999999999999999 2 0 1\n",
        "SUBMIT a 0 1 2 0 1 0 garbage\nx\n", "CANCEL active garbage\n",
        "SUBMIT a 0 1 2 0 1", "SUBMIT a 0 1 2 0 1 4\nxabc"
    };
    for (size_t i=0; i<sizeof(bad)/sizeof(bad[0]); i++) {
        text_input(bad[i]); CHECK(serve_read_cmd("active")==-2,"malformed frame is fatal");
        CHECK(serve_read_cmd("active")==-2,"fatal input remains unusable");
    }
    char long_header[600]; memset(long_header,'x',sizeof(long_header)); long_header[599]='\n';
    input(long_header,sizeof(long_header)); CHECK(serve_read_cmd("active")==-2,"overlong header fatal");
    output(out,sizeof(out)); CHECK(strstr(out,"ERROR active") == NULL,"parser does not duplicate active terminal");
    fclose(wire);
}
static void transaction_boundaries(void) {
    char out[4096]; float one=1; char *names[]={"x"}; int added[]={1};
    Special special={"x",1,0}; Tok t={0};
    t.sp=&special; t.nsp=1; t.id2str=names; t.id_added=added; t.n_ids=1;
    Model m={0}; m.c.hidden=1; m.c.unpad_vocab=1; m.c.eps=1e-6f;
    m.c.eos=-1; m.c.mel_bins=80; m.c.mup=1;
    m.embed.f=&one; m.final_norm=&one; m.lm_head.f=&one;
    SReq q={0}; strcpy(q.id,"active"); q.payload="x"; q.plen=1; q.max_tok=2;
    capture(); text_input("");
    serve_one(&m,&t,&q); output(out,sizeof(out));
    CHECK(strstr(out,"ACCEPT active 1\n") && strstr(out,"DONE active STAT 2 "),"EOF finishes accepted request");
    CHECK(m.kv_len==2 && m.kvp.len==2,"only computed tokens enter prefix");
    fclose(wire);
    capture(); text_input(""); q.stop_requested=1;
    serve_one(&m,&t,&q); output(out,sizeof(out));
    CHECK(strstr(out,"ACCEPT active 1\n") && strstr(out,"DONE active STAT 0 "),"queued STOP accepted and completed");
    CHECK(!strstr(out,"DATA ") && m.kvp.len==2,"queued STOP performs no forward");
    fclose(wire);
    capture(); text_input("STOP active\nCANCEL active\n"); q.stop_requested=0;
    serve_one(&m,&t,&q); output(out,sizeof(out));
    CHECK(strstr(out,"ERROR active CANCELLED\n") && !strstr(out,"DONE "),"CANCEL beats STOP");
    CHECK(m.kvp.len==2,"pre-forward cancellation preserves computed prefix");
    fclose(wire);
    capture(); text_input("SUBMIT queued 0 1 2 0 1\nx\nCANCEL active garbage\n");
    serve_one(&m,&t,&q); output(out,sizeof(out));
    CHECK(g_qn==0 && g_input_bad,"fatal control frees queued ownership");
    CHECK(strstr(out,"ERROR active BAD_FRAME\n") && !strstr(strstr(out,"ERROR active BAD_FRAME\n")+1,"ERROR active"),"exactly one active fatal terminal");
    CHECK(!strstr(out,"DONE "),"fatal transaction never succeeds"); fclose(wire);
    capture(); boundary_input("ACCEPT %s %d\n","CANCEL active\n");
    m.kv_len=7;
    serve_one(&m,&t,&q); output(out,sizeof(out)); close(control_fd); control_fd=-1;
    CHECK(m.kv_len==7 && !strstr(out,"DATA "),"ACCEPT-boundary CANCEL prevents prefill");
    CHECK(strstr(out,"ERROR active CANCELLED\n") && !strstr(out,"DONE "),"ACCEPT-boundary cancellation terminal"); fclose(wire);
    capture(); boundary_input("DATA %s %d\n","STOP active\nCANCEL active\n");
    serve_one(&m,&t,&q); output(out,sizeof(out)); close(control_fd); control_fd=-1;
    char *data=strstr(out,"DATA active");
    CHECK(data && !strstr(data+1,"DATA active"),"DATA-boundary controls prevent following token");
    CHECK(strstr(out,"ERROR active CANCELLED\n") && !strstr(out,"DONE "),"DATA-boundary CANCEL wins"); fclose(wire);
    CHECK(m.kvp.len==1,"cancelled emitted token is not recorded as computed");
    capture(); boundary_input("DATA %s %d\n","STOP active\n");
    serve_one(&m,&t,&q); output(out,sizeof(out)); close(control_fd); control_fd=-1;
    data=strstr(out,"DATA active");
    CHECK(data && !strstr(data+1,"DATA active"),"STOP prevents further DATA");
    CHECK(strstr(out,"DONE active STAT 1 ") && strstr(out," 1 0\n") && !strstr(out,"ERROR "),"active STOP is successful and not length limited");
    fclose(wire);
    capture(); text_input("CANCEL old\n"); strcpy(q.id,"next");
    serve_one(&m,&t,&q); output(out,sizeof(out));
    CHECK(strstr(out,"ERROR old NOT_FOUND\n") && strstr(out,"DONE next STAT 2 "),"late CANCEL cannot affect following request");
    fclose(wire);
    char *old_limit=getenv("CTX_MAX"); char *saved_limit=old_limit ? strdup(old_limit) : NULL;
    setenv("CTX_MAX","2",1); capture(); text_input("");
    serve_one(&m,&t,&q); output(out,sizeof(out));
    CHECK(strstr(out,"ERROR next CONTEXT_EXCEEDED 3 2\n") && !strstr(out,"ACCEPT "),"context rejection is positional and pre-accept");
    fclose(wire);
    if (saved_limit) { setenv("CTX_MAX",saved_limit,1); free(saved_limit); } else unsetenv("CTX_MAX");
    free(m.K); free(m.V); kv_prefix_free(&m.kvp);
}

static void fragmented_payload(void) {
    const char frame[]="SUBMIT unicode 0 7 2 0 1 3\n\xc3\xa9\n\xf0\x9f\x99\x82\x00\xff\n\n";
    int fd[2]; CHECK(pipe(fd)==0,"fragment pipe");
    pid_t child=fork();
    if (child==0) { close(fd[0]); for (size_t i=0;i<sizeof(frame)-1;i++) if(write(fd[1],frame+i,1)!=1) _exit(1); close(fd[1]); _exit(0); }
    close(fd[1]); dup2(fd[0],fileno(stdin)); close(fd[0]); clearerr(stdin); g_input_bad=g_input_closed=0;
    capture(); CHECK(serve_read_cmd(NULL)==0 && g_qn==1,"fragmented Unicode/audio frame accepted");
    CHECK(g_qn && g_q[0].plen==7 && !memcmp(g_q[0].payload,"\xc3\xa9\n\xf0\x9f\x99\x82",7),"Unicode and embedded newline preserved");
    CHECK(g_qn && g_q[0].alen==3 && !memcmp(g_q[0].audio,"\x00\xff\n",3),"binary audio preserved");
    CHECK(serve_read_cmd(NULL)==-1,"fragmented frame EOF");
    int status; waitpid(child,&status,0); CHECK(WIFEXITED(status) && !WEXITSTATUS(status),"fragment writer");
    fclose(wire); clear_queue();
    size_t alen=(1<<20)+1; size_t cap=alen+100; char *large=malloc(cap);
    int n=snprintf(large,cap,"SUBMIT large 0 1 2 0 1 %zu\nx",alen);
    memset(large+n,0,alen); large[n+alen]='\n';
    capture(); input(large,(size_t)n+alen+1); free(large);
    CHECK(serve_read_cmd(NULL)==0 && g_qn==1 && g_q[0].alen==(int)alen,"audio larger than generic 1MiB ceiling accepted");
    fclose(wire); clear_queue();
}
int main(void) {
    setvbuf(stdin,NULL,_IONBF,0); int saved=dup(fileno(stdout));
    controls(); framing(); transaction_boundaries(); fragmented_payload();
    fflush(stdout); dup2(saved,fileno(stdout)); close(saved);
    if (failures) return 1;
    puts("inkling serve control: ok"); return 0;
}
