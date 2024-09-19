/*lex.l*/
%{
#include <stdio.h>
#include "parser.tab.h"
void yyerror(char *);
%}

%option noyywrap
NUM [1-9]+[0-9]*|0
ID [a-zA-Z_][a-zA-Z0-9_]*

%%
{NUM}   { yylval.inum=atoi(yytext); return NUM; }
"print" { return PRINT; }
"int"   { return INT; }
{ID}    { yylval.str=strdup(yytext); return ID; }
"+"     { return ADD; }
"-"     { return SUB; }
"*"     { return MUL; }
"/"     { return DIV; }
"("     { return LP; }
")"     { return RP; }
\n      { return NEWLINE; }
"="     { return ASSIGN; }
";"     { return SEMICOLON; }
","     { return COMMA; }
[ \t]+  /* ignore whitespace */;
.       
%%
