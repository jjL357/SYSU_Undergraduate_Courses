/*parser2.y*/
%{
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int yylex(void);
void yyerror(char *str);

typedef struct Variables{
    char* var;
    int value;
    int is_assigned;
    struct Variables* next;
}Variables;

Variables *head = NULL;

Variables* Variables_find(const char*var_name);
void Variables_add(const char*var_name, int value,int is_assigned);
void Variables_free();

%}

%union{
  int inum;
  double dnum;
  char* name;
}

%token ADD SUB MUL DIV NEWLINE SEMICOLON ASSIGN COMMA PRINT INT R_BRACKET L_BRACKET
%token <inum> NUM
%token <name> VAR
%type <inum> expression term single
%%
        code: 
            | code line;

        line :   codes_in_line NEWLINE;

        codes_in_line: code_in_line
            |codes_in_line code_in_line ;

        code_in_line :SEMICOLON
                | definition SEMICOLON 
                
                | PRINT L_BRACKET expression R_BRACKET SEMICOLON{
                    printf("%d\n",$3);
                }
                | var_assign SEMICOLON;
                


        var_assign: VAR ASSIGN expression {
            Variables*temp = Variables_find($1);
            if(!temp){
                char msg[256];
                sprintf(msg, "%s", $1);
                yyerror(msg);
            }
            else {
                temp->value = $3;
                temp->is_assigned = 1;
            }
        }  
        | VAR ASSIGN expression COMMA var_assign;

        definition: INT VAR ASSIGN expression  {Variables_add($2,$4,1);}
                | INT VAR  {Variables_add($2,0,0);}
                | definition COMMA VAR  {Variables_add($3,0,0);}
                | definition COMMA VAR ASSIGN expression  {Variables_add($3,$5,1);};

        expression: term
                | expression ADD term { $$ = $1 + $3; }
                | expression SUB term { $$ = $1 - $3; };

        term: single
                | term MUL single { $$ = $1 * $3; }
                | term DIV single { $$ = $1 / $3; };

        single: NUM
                | L_BRACKET expression R_BRACKET{$$ = $2;}
                | SUB NUM {$$ = -$2;}
                | VAR {
                    Variables *find = Variables_find($1);
                    if(!find){
                        
                        char msg[256];
                        sprintf(msg, "%s", $1);
                        yyerror(msg);
                    }
                    else {
                        $$ = find->value;
                    }
                };
                

                
%%
void yyerror(char * str){
    
    printf("Undefined Variable %s\n",str);
    exit(EXIT_FAILURE);
}

Variables* Variables_find(const char*var_name){
    Variables* temp = head;
    while(temp){
        if(!strcmp(temp->var, var_name)){
            return temp;
        }
        temp = temp->next;
    }
    return NULL;
}

void Variables_add(const char*var_name, int value,int is_assigned){
    Variables*new_var = (Variables*)malloc(sizeof(Variables));
    new_var->value = value;
    new_var->var = (char*)malloc(strlen(var_name)+1);
    new_var->is_assigned = is_assigned;
    strcpy(new_var->var,var_name);
    new_var->next = head;
    head = new_var;
    
}

void Variables_free(){
    Variables* temp =  head;
    while(temp){
        Variables*temp_next = temp->next;
        free(temp->var);
        free(temp);
        temp = temp_next;
    }
    head = NULL;
}

int main()
{
    yyparse();
    Variables_free();
}
