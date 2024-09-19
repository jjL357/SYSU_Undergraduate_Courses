/*lex.l*/
%{
#include <stdio.h>
#include "parser.tab.h"
void yyerror(char *);
%}
%option noyywrap
NUM [1-9]+[0-9]*|0
%%
{NUM}   {yylval.inum=atoi(yytext);return NUM;}
"+"     return ADD;
"-"     return SUB;
"*"     return MUL;
"/"     return DIV;
\n      return NEWLINE;
[ \t]+      /* ignore whitespace */;
.
%%