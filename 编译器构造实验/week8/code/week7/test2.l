%option noyywrap

%%

float[ \t]+   { printf("double "); }


\n | .     { printf("%s", yytext); }
\"[^\"\n]*\"  { printf("%s", yytext); }



%%

int main() {
    yylex();
    return 0;
}
