/* Include the production generation unit, not a duplicate protocol parser. */
#define COLI_V4_UNIT_GENERATE_STATS
#define COLI_V4_SKIP_GENERATE_MAIN
#include "../deepseek_v4.c"
#include <assert.h>
#ifndef _WIN32
#include <sys/wait.h>
#endif

static int failures;
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c); failures++; } } while (0)
static FILE *capture;
static int saved_out, saved_in;
static void input(const void *bytes,size_t n) {
    FILE *f=tmpfile(); assert(f); assert(fwrite(bytes,1,n,f)==n); rewind(f);
    v4_serve_input_closed=0;
    assert(dup2(fileno(f),fileno(stdin))>=0); fclose(f); clearerr(stdin);
}
static void text_input(const char *s) { input(s,strlen(s)); }
static void reset_output(void) {
    fflush(stdout); assert(ftruncate(fileno(capture),0)==0); rewind(capture);
}
static char *wire(void) {
    static char b[8192]; fflush(stdout); rewind(capture);
    size_t n=fread(b,1,sizeof(b)-1,capture); b[n]=0; fseek(capture,0,SEEK_END); return b;
}
static void terminal_tests(void) {
    reset_output(); v4_serve_emit_terminal("A",3,2,1,2,3,4,1,7);
    CHECK(!strcmp(wire(),"ERROR A CANCELLED\n"));
    reset_output(); v4_serve_emit_terminal("A",1,2,1,2,3,4,1,7);
    CHECK(strstr(wire(),"DONE A STAT 2 1.000 2.0 3.00 4 0 7\n")!=NULL);
    reset_output(); v4_serve_emit_terminal("A",-2,2,1,2,3,4,1,7);
    CHECK(!strcmp(wire(),"ERROR A BAD_FRAME\n"));
    reset_output(); v4_serve_emit_terminal("B",0,2,1,2,3,4,1,7);
    CHECK(strstr(wire(),"DONE B STAT 2 1.000 2.0 3.00 4 1 7\n")!=NULL);
}
static void control_tests(void) {
    V4ServeRequest q={0};
    text_input("STOP A\nCANCEL A\n");
    CHECK(v4_serve_read_request(&q,"A")==1); CHECK(v4_serve_read_request(&q,"A")==3);
    reset_output(); text_input("CANCEL old\nSTOP old\n");
    CHECK(v4_serve_read_request(&q,"B")==0); CHECK(v4_serve_read_request(&q,"B")==0);
    CHECK(!strcmp(wire(),"ERROR old NOT_FOUND\nERROR old NOT_FOUND\n"));
    text_input("SUBMIT B 0 1 2 0 1\nx\n");
    CHECK(v4_serve_read_request(&q,"A")==2); CHECK(q.prompt && !strcmp(q.prompt,"x")); free(q.prompt);
    CHECK(v4_serve_read_request(&q,"A")==-1);
    reset_output(); v4_serve_input_closed=0;
    text_input("STOP A\nSUBMIT B 0 1 2 0 1\nx\nCANCEL A\n");
    V4ServeStream stream={.request_id="A"};
    CHECK(v4_serve_poll_controls(&stream)==3);
    CHECK(!strcmp(wire(),"ERROR B engine busy\n")); CHECK(stream.input_closed);
    CHECK(v4_serve_poll_controls(&stream)==3);
    reset_output(); v4_serve_input_closed=0; text_input("CANCEL A\n");
    stream=(V4ServeStream){.request_id="A"};
    /* Poll at callback entry: even an unusable tokenizer is never touched. */
    CHECK(v4_serve_token(&stream,99,0,0,1)==1);
    CHECK(stream.control==3); CHECK(!strcmp(wire(),""));
    reset_output(); v4_serve_input_closed=0; text_input("");
    stream=(V4ServeStream){.request_id="B"};
    CHECK(v4_serve_token(&stream,1,0,0,1)==0);
    CHECK(stream.input_closed); CHECK(stream.control==0);
}
static void framing_tests(void) {
    V4ServeRequest q={0};
    reset_output(); text_input("SUBMIT R 7 9 0 0 1\nCANCEL A\n\nSUBMIT B 0 1 2 0 1\nx\n");
    CHECK(v4_serve_read_request(&q,"A")==0);
    int r=v4_serve_read_request(&q,"A"); CHECK(r==2);
    if(r==2){ CHECK(!strcmp(q.id,"B")); free(q.prompt); }
    CHECK(strstr(wire(),"ERROR A")==NULL);
    reset_output(); text_input("SUBMIT R 0 1 2 0 1 9\nxCANCEL A\n\nSUBMIT B 0 1 2 0 1\nx\n");
    CHECK(v4_serve_read_request(&q,"A")==0);
    r=v4_serve_read_request(&q,"A"); CHECK(r==2);
    if(r==2){ CHECK(!strcmp(q.id,"B")); free(q.prompt); }
    CHECK(strstr(wire(),"ERROR A")==NULL);
    const char payload[]="é\n界";
    char frame[128]; int n=snprintf(frame,sizeof(frame),"SUBMIT U 0 %zu 2 0 1 0\n",strlen(payload));
    memcpy(frame+n,payload,strlen(payload)); n+=(int)strlen(payload); frame[n++]='\n';
    input(frame,(size_t)n); CHECK(v4_serve_read_request(&q,NULL)==2);
    CHECK(q.prompt_bytes==(int)strlen(payload)); CHECK(!memcmp(q.prompt,payload,strlen(payload))); free(q.prompt);
    const char *bad[]={"SUBMIT A 0 1 2 0 1\nx!CANCEL A\n",
        "SUBMIT A 0 3 2 0 1\nx", "SUBMIT A 0 -1 2 0 1\n",
        "SUBMIT A 0 99999999999999999999999 2 0 1\n",
        "SUBMIT A 0 1 2 0 1 trailing\nx\n", "STOP A junk\n", "CANCEL A"};
    for(size_t i=0;i<sizeof(bad)/sizeof(*bad);i++){
        reset_output(); text_input(bad[i]);
        CHECK(v4_serve_read_request(&q,"A")==-2); CHECK(v4_serve_input_closed);
        CHECK(!strcmp(wire(),""));
    }
    char long_header[600]; memset(long_header,'x',sizeof(long_header));
    memcpy(long_header,"SUBMIT A ",9); long_header[599]='\n';
    reset_output(); input(long_header,sizeof(long_header));
    CHECK(v4_serve_read_request(&q,"A")==-2); CHECK(v4_serve_input_closed);
    reset_output(); text_input("SUBMIT A 0 1 2 0 1\nx!CANCEL A\n");
    V4ServeStream stream={.request_id="A"};
    CHECK(v4_serve_poll_controls(&stream)==-2);
    v4_serve_emit_terminal("A",stream.control,0,0,0,0,1,0,0);
    CHECK(!strcmp(wire(),"ERROR A BAD_FRAME\n"));
}

static void fragmented_input_test(void) {
#ifndef _WIN32
    const char frame[]="SUBMIT U 0 7 2 0 1\n\xc3\xa9\n\xf0\x9f\x99\x82\n";
    int fd[2]; assert(pipe(fd)==0);
    pid_t child=fork(); assert(child>=0);
    if(!child){
        close(fd[0]);
        for(size_t i=0;i<sizeof(frame)-1;i++)
            if(write(fd[1],frame+i,1)!=1) _exit(1);
        close(fd[1]); _exit(0);
    }
    close(fd[1]); assert(dup2(fd[0],fileno(stdin))>=0); close(fd[0]);
    clearerr(stdin); v4_serve_input_closed=0;
    V4ServeRequest q={0}; reset_output();
    CHECK(v4_serve_read_request(&q,NULL)==2);
    CHECK(q.prompt_bytes==7); CHECK(!memcmp(q.prompt,"\xc3\xa9\n\xf0\x9f\x99\x82",7));
    free(q.prompt);
    CHECK(v4_serve_read_request(&q,NULL)==-1); CHECK(!strcmp(wire(),""));
    int status=0; CHECK(waitpid(child,&status,0)==child);
    CHECK(WIFEXITED(status)&&WEXITSTATUS(status)==0);
#endif
}

static void accepted_control_tests(void) {
    ColiV4Engine engine={0}; ColiV4Session session={0};
    int prompt_ids[32]={0};
    Special special={.str="X",.len=1,.id=2};
    engine.runtime.context_tokens=8;
    session.max_prompt_tokens=8; session.prompt_ids=prompt_ids;
    session.tokenizer.sp=&special; session.tokenizer.nsp=1;
    V4ServeRequest q={.id="A",.prompt="X",.prompt_bytes=1,
                      .max_tokens=1,.temperature=0,.top_p=1};
    reset_output(); text_input("STOP A\n");
    v4_serve_one(&engine,&session,&q);
    CHECK(strstr(wire(),"ACCEPT A 1\nDONE A STAT 0 ")!=NULL);
    CHECK(strstr(wire()," 1 0 0\n")!=NULL);
    reset_output(); text_input("STOP A\nCANCEL A\n");
    v4_serve_one(&engine,&session,&q);
    CHECK(!strcmp(wire(),"ACCEPT A 1\nERROR A CANCELLED\n"));
    reset_output(); text_input("SUBMIT A 0 1 1 0 1\nx!");
    v4_serve_one(&engine,&session,&q);
    CHECK(!strcmp(wire(),"ACCEPT A 1\nERROR A BAD_FRAME\n"));
}

int main(void) {
    saved_out=dup(fileno(stdout)); saved_in=dup(fileno(stdin)); capture=tmpfile(); assert(capture);
    fflush(stdout); assert(dup2(fileno(capture),fileno(stdout))>=0); setvbuf(stdin,NULL,_IONBF,0);
    terminal_tests(); control_tests(); framing_tests(); fragmented_input_test(); accepted_control_tests();
    fflush(stdout); dup2(saved_out,fileno(stdout)); dup2(saved_in,fileno(stdin)); close(saved_out); close(saved_in); fclose(capture);
    if(failures) return 1; puts("v4 serve control: ok"); return 0;
}
