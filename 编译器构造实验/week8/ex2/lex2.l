/*lex2.l*/
%{
#include <stdio.h>
#include "parser2.tab.h"
void yyerror(char *);
%}

%option noyywrap
NUM [1-9]+[0-9]*|0
VAR [a-zA-Z_][a-zA-Z0-9_]*

%%
{NUM}   {yylval.inum=atoi(yytext);return NUM;}
"print" return PRINT; 
"int"   return INT; 
{VAR}   {yylval.name=strdup(yytext); return VAR;}
"("     return L_BRACKET;
")"     return R_BRACKET;
"="     return ASSIGN;
";"     return SEMICOLON;
","     return COMMA;
"+"     return ADD;
"-"     return SUB;
"*"     return MUL;
"/"     return DIV;
\n      return NEWLINE;
[ \t]+      /* ignore whitespace */;
.
%%