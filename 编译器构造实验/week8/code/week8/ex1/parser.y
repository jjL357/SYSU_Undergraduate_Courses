/*parser.y*/
%{
#include <stdio.h>
int yylex(void);
void yyerror(char *);
%}

%union{
  int inum;
  double dnum;
}

%token ADD SUB MUL DIV NEWLINE
%token <inum> NUM
%type <inum> expression term single
%%
        line_list: line
                | line_list line;
        line : expression NEWLINE {printf("#ans is %d\n",$1);}
                | error NEWLINE { yyerrok;}
        expression: term
                | expression ADD term { $$ = $1 + $3; }
                | expression SUB term { $$ = $1 - $3; };
        term: single
                | term MUL single { $$ = $1 * $3; }
                | term DIV single { $$ = $1 / $3; };
        single: NUM;
%%
void yyerror(char * str){
    printf("Invalid\n");
}

int main()
{
    yyparse();
}
