/*parser.y*/
%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Variable{
    char* name;
    int value;
    struct Variable *next;
} Variable;

int yylex(void);
void yyerror(char *);
void add_variable(const char *name, int value);
void free_variables();
Variable *find_variable(const char *name);
Variable *variables = NULL;
%}

%union{
    int inum;
    double dnum;
    char* str;
}

%token ADD SUB MUL DIV NEWLINE LP RP SEMICOLON ASSIGN COMMA PRINT INT
%token <inum> NUM
%token <str> ID
%type <inum> expression term single

%%
program:
    | program line
    ;
line: sentences NEWLINE
    ;
sentences: sentence
    | sentences sentence;
sentence: SEMICOLON
    | definition SEMICOLON
    | PRINT LP expression RP SEMICOLON { 
        printf("%d\n", $3);
    } 
    | ID ASSIGN expression SEMICOLON {
        Variable *var = find_variable($1);
         if (var == NULL) {
            char error_msg[256];
            sprintf(error_msg, "Undefined Variable %s", $1);
            yyerror(error_msg);
        }
        else 
            var->value = $3;
    };

definition:
    INT ID ASSIGN expression { add_variable($2, $4); }
    | INT ID { add_variable($2, 0); }
    | definition COMMA ID { add_variable($3, 0); }
    | definition COMMA ID ASSIGN expression { add_variable($3, $5); };
expression: term
    | expression ADD term { $$ = $1 + $3; }
    | expression SUB term { $$ = $1 - $3; };
term: single
    | term MUL single { $$ = $1 * $3; }
    | term DIV single { $$ = $1 / $3; };
single: NUM
    | LP expression RP {$$ = $2;}
    | SUB NUM {$$ = -$2;}
    | ID { 
        Variable *var = find_variable($1); 
        if (var == NULL) { 
            char error_msg[256];
            sprintf(error_msg, "Undefined Variable %s", $1);
            yyerror(error_msg);
        } 
        else
            $$ = var->value; 
    };
%%

void add_variable(const char *name, int value) {
    Variable *new_variable = (Variable *)malloc(sizeof(Variable));
    new_variable->name = (char *)malloc(strlen(name) + 1);

    strcpy(new_variable->name, name);
    new_variable->value = value;
    new_variable->next = variables;
    variables = new_variable;
}

Variable *find_variable(const char *name) {
    Variable *current = variables;
    while (current != NULL) {
        if (strcmp(current->name, name) == 0) {
            return current;
        }
        current = current->next;
    }
    return NULL;
}

void free_variables() {
    Variable *current = variables;
    while (current != NULL) {
        Variable *next = current->next;
        free(current->name);
        free(current);
        current = next;
    }
    variables = NULL;
}

void yyerror(char * str){
    printf("%s\n", str);
    exit(EXIT_FAILURE);
}

int main()
{
    yyparse();
    free_variables();
}