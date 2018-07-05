//
//  Parsing.c
//  FeedforwardNT
//
//  Created by Hakime Seddik on 22/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Parsing.h"
#include "NeuralNetwork.h"


definition * _Nonnull allocateDefinitionNode(void) {
    
    definition *r = (definition *)malloc(sizeof(definition));
    *r = (definition){.def_id=0, .number_of_fields=0, .field=NULL, .next=NULL, .previous=NULL};
    return r;
}

dictionary * _Nonnull allocateDictionaryNode(void) {
    
    dictionary *d = (dictionary *)malloc(sizeof(dictionary));
    *d = (dictionary){.has_tag=false, .next=NULL, .previous=NULL};
    bzero(d->key, MAX_LONG_STRING_LENGTH);
    bzero(d->value, MAX_LONG_STRING_LENGTH);
    bzero(d->tag, 1);
    return d;
}

//
// Get the network parameters definitions
//
definition * _Nullable getDefinitions(void * _Nonnull neural, const char * _Nonnull paramsDefFile, const char * _Nonnull keyword) {
    
    NeuralNetwork *nn = (NeuralNetwork *)neural;
    
    definition *definitions = NULL;
    
    FILE *f1 = fopen(paramsDefFile,"r");
    if(!f1) {
        fprintf(stdout,"%s: can't find the parameters definition file %s.\n", PROGRAM_NAME, paramsDefFile);
        return NULL;
    }
    
    int lineCount = 1;
    bool first_character = true, found_tag = false;
    char ch = 0;
    char str[MAX_LONG_STRING_LENGTH];
    char tag[1];
    bool new_def;
    char buff[MAX_LONG_STRING_LENGTH];
    int idx = 0;
    int total_key_values = 0;
    int defID = 0;
    bool first_d_node = true;
    definition *d_head = NULL, *d_pos = NULL, *d_pt = NULL;
    dictionary *field_head = NULL, *field_pos = NULL, *field_pt = NULL;
    while(1) {
        ch = fgetc(f1);
        if (ch == -1) {
            fprintf(stderr, "%s: syntax error in the file %s. File should end with <}>.\n", PROGRAM_NAME, paramsDefFile);
            fatal(PROGRAM_NAME);
        }
        if (ch == ' ') continue;
        if (first_character && ch != '{') { // First character in file should be {
            fprintf(stderr, "%s: syntax error in the file %s. File should start with <{>.\n", PROGRAM_NAME, paramsDefFile);
            fatal(PROGRAM_NAME);
        } else if (first_character && ch == '{'){
            first_character = false;
            continue;
        }
        if (ch == '}') { // End of file
            break;
        }
        
        if(ch == '\n'){
            lineCount++;
        } else {
            if (idx > MAX_LONG_STRING_LENGTH) {
                fatal(PROGRAM_NAME, "string larger than buffer in getDefinitions().");
            }
            buff[idx] = ch;
            idx++;
            if (ch == '{' && !first_character) {
                bool alreadyCanonized = false;
                memset(str, 0, sizeof(str));
                memcpy(str, buff, strlen(keyword));
                if (strcmp(str, keyword) != 0) {
                    fatal(PROGRAM_NAME, "incorrect keyword for parameter definition. Should be:", (char *)keyword);
                }
                // New parameter definition starts here
                nn->number_of_parameters++;
                memset(buff, 0, sizeof(buff));
                memset(tag, 0, sizeof(tag));
                idx = 0;
                if (first_d_node) {
                    d_head = allocateDefinitionNode();
                    d_head->def_id = defID;
                    definitions = d_head;
                    d_pos = d_head;
                    d_pt = d_head;
                    defID++;
                } else {
                    d_pt = allocateDefinitionNode();
                    d_pt->def_id = defID;
                    defID++;
                }
                new_def = true;
                bool field_line = false;
                bool first_kv_node = true;
                int found_key = 0;
                while(1) {
                    ch = fgetc(f1);
                    if (ch == ' ') {
                        // A space in the key name is replaced internally by '_'
                        // (NB: Spaces in the key value are just ignored)
                        if (found_key == 0 && buff[0] != 0 && !alreadyCanonized) {
                            buff[idx] = '_';
                            idx++;
                            alreadyCanonized = true;
                        }
                        continue;
                    }
                    if (ch == '}' && new_def) { // End of definition
                        if (!first_d_node) {
                            d_pos->next = d_pt;
                            d_pt->previous = d_pos;
                            d_pos = d_pt;
                        }
                        total_key_values += d_pt->number_of_fields;
                        new_def = false;
                        first_d_node = false;
                        break;
                    }
                    if(ch == '\n'){
                        if (!field_line) {
                            lineCount++;
                            field_line = true;
                            continue;
                        }
                        memcpy(field_pt->value, buff, idx);
                        if (found_tag) {
                            memcpy(field_pt->tag, tag, 1);
                            field_pt->has_tag = true;
                        }
                        if (!first_kv_node) {
                            field_pos->next = field_pt;
                            field_pt->previous = field_pos;
                            field_pos = field_pt;
                        }
                        d_pt->number_of_fields++;
                        first_kv_node = false;
                        memset(buff, 0, sizeof(buff));
                        idx = 0;
                        lineCount++;
                        found_key = 0;
                        found_tag = false;
                        memset(tag, 0, sizeof(tag));
                    } else {
                        if (ch == ':') {
                            found_key++;
                            if (found_key > 1) {
                                fprintf(stderr, "%s: syntax error in the file %s. Maybe duplicate character <:>.\n", PROGRAM_NAME, paramsDefFile);
                                fatal(PROGRAM_NAME);
                            }
                            if (first_kv_node) {
                                field_head = allocateDictionaryNode();
                                d_pt->field = field_head;
                                field_pos = field_head;
                                field_pt = field_head;
                            } else {
                                field_pt = allocateDictionaryNode();
                            }
                            memcpy(field_pt->key, buff, idx);
                            memset(buff, 0, sizeof(buff));
                            idx = 0;
                        } else if (ch == '<' || ch == '=' || ch == '>') {
                            found_tag = true;
                            if (ch == '<') memcpy(tag, "<", 1);
                            if (ch == '=') memcpy(tag, "=", 1);
                            if (ch == '>') memcpy(tag, ">", 1);
                        } else {
                            buff[idx] = ch;
                            idx++;
                            if (idx >= MAX_LONG_STRING_LENGTH) {
                                fatal(PROGRAM_NAME, "string larger than buffer in getDefinitions().");
                            }
                        }
                    }
                }
            }
        }
    }
    if (total_key_values == 0) {
        fatal(PROGRAM_NAME, "no network parameters found in definition input file!");
    }
    
    return definitions;
}


