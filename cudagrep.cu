#include <dirent.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

using namespace std;

void sendGPU(char *A, unsigned int indexOfLine[], unsigned int &iLine); //send to GPU routine

char *d_word; //GPU MEMORY word pattern pointer.
char word[100];


// EDIT THIS ACCORDING TO YOUR GPU CAPABILITIES
const int MAX_THREADS = 1024;
const int MAX_BLOCKS = 1024;

// GREP GPU KERNEL
__global__ void GrepKernel(char *A, bool *R, char *wordD, unsigned int *indexOfLine)
{
    int row = threadIdx.x + blockIdx.x * gridDim.x;
    
    if(indexOfLine[row+2]!=0) //To be sure that there is something to check.
    {
        int indexWord = 0;
        bool matching = false;
        bool matched = false;
        
        bool starting = false; //^ REGEX
        if(wordD[0]=='^')
        {
            indexWord=1;
            matching = true;
            starting = true;
        }
        
        for(int j=0;j<(indexOfLine[row+1]-indexOfLine[row]);j++) //Check each characters
        {
            if(A[indexOfLine[row]+j]=='\0')//End of the line reached
                break;
            
            if(((A[indexOfLine[row]+j]==wordD[indexWord])&&(matching==true || indexWord==0))||(wordD[indexWord]=='.'))//Letter match
            {
                matching=true;
                indexWord++;
                if(wordD[indexWord]=='\0')
                {
                    matched = true;
                    break;
                }
            }
            else if(matching==true)//Was matching and letter does not seem to match
            {
                if(wordD[indexWord]=='$') //$ REGEX
                {
                    if(j+1==(indexOfLine[row+1]-indexOfLine[row]))
                    {
                        matched=true;
                        break;
                    }
                    else //It does not match anymore RESTART matching
                    {
                        indexWord=0;
                        matching=false;
                    }
                }
                else if(wordD[indexWord]=='*' && A[indexOfLine[row]+j]!=' ')//* REGEX
                {
                    if(wordD[indexWord+1]==A[indexOfLine[row]+j+1])
                    {
                        indexWord++;
                    }
                }
                else if(starting) //^REGEX
                    break;
                else //It does not match anymore RESTART matching
                {
                    indexWord=0;
                    matching=false;
                }
            }
        }
        
        if(matched)
            R[row]=true;
        else
            R[row]=false;
    }
    
}

int main(int argc, const char * argv[])
{
    strcpy(word, argv[2]);
    
    //LOAD word INTO DEVICE MEMORY
    cudaMalloc((void**)&d_word, 100);
    cudaMemcpy(d_word, word, 100, cudaMemcpyHostToDevice);
    
    //ALLOC ARRAY
    char *A=(char*)malloc(2000000000);
    A[0]='\0';
    
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    
    //OPEN FILE
    fp = fopen(argv[1], "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    
    unsigned int indexOfLine[(MAX_THREADS*MAX_BLOCKS)+2];
    indexOfLine[0]=0;
    
    unsigned int iLine = 0;
    
    while ((read = getline(&line, &len, fp)) != -1)//Line per line read
    {
        int index = indexOfLine[iLine];
        int i=0;
        for(i;i>-1;i++)//Fill the array
        {
            if(line[i]=='\0')
                break;
            
            A[index+i]=line[i];
        }
        iLine++;
        
        indexOfLine[iLine]=i+index; //Store the index of the started line.
        
        if(iLine>=MAX_THREADS*MAX_BLOCKS)//MAX Amount of lines reached so send to the GPU
        {
            indexOfLine[iLine+1]=2;
            sendGPU(A,indexOfLine,iLine);
        }
    }
    
    
    if (line)
        free(line);
    
    //File fully read, last send to the GPU
    for(int i=iLine+1;i<(MAX_BLOCKS*MAX_THREADS)+2;i++)
        indexOfLine[i]=0;
    
    sendGPU(A, indexOfLine,iLine);
    
    //FREE the memory
    cudaFree(d_word);
    free(A);
    
    return 0;
}

void sendGPU(char *A, unsigned int indexOfLine[], unsigned int &iLine)
{
    //Allocations
    unsigned int size = indexOfLine[iLine];
    bool *R=(bool*)malloc(iLine*sizeof(bool));
    
    //SEND TO GPU ROUTINE
    
    //LOAD A INTO DEVICE MEMORY
    char *d_A;
    cudaError_t err = cudaMalloc((void**)&d_A, size);
    //printf("CUDA malloc A: %s\n",cudaGetErrorString(err));
    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    //printf("Copy A to device: %s\n",cudaGetErrorString(err));
    
    //LOAD indexOfLine INTO DEVICE MEMORY
    unsigned int *d_indexOfLine;
    err = cudaMalloc((void**)&d_indexOfLine, ((MAX_THREADS*MAX_BLOCKS)+2)*sizeof(unsigned int));
    err = cudaMemcpy(d_indexOfLine, indexOfLine, ((MAX_THREADS*MAX_BLOCKS)+2)*sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    //CREATE R FOR RESULTS
    bool *d_R;
    err = cudaMalloc((void**)&d_R, iLine*sizeof(bool));
    //printf("CUDA malloc C: %s\n",cudaGetErrorString(err));
    
    // Invoke kernel
    dim3 dimBlock(MAX_THREADS,1);
    dim3 dimGrid(MAX_BLOCKS,1);
    GrepKernel<<<dimGrid, dimBlock>>>(d_A, d_R, d_word, d_indexOfLine);
    
    //Wait that the GPU work is over.
    err = cudaThreadSynchronize();
    
    
    //printf("Run kernel: %s\n", cudaGetErrorString(err));
    
    // Read R from device memory
    err = cudaMemcpy(R, d_R, iLine*sizeof(bool), cudaMemcpyDeviceToHost);
    //printf("Copy R off of device: %s\n",cudaGetErrorString(err));
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_R);
    cudaFree(d_indexOfLine);
    
    // Display matched lines
    for(int i=0;i<iLine;i++)
    {
        if(R[i])
        {
            for(int j=0;j<(indexOfLine[i+1]-indexOfLine[i]);j++)
            {
                char letter = A[indexOfLine[i]+j];
                if(letter=='\0')
                    break;
                else
                    printf("%c",letter);
            }
            
        }
    }
    
    
    
    //Reset memory and counter.
    free(R);
    iLine=0;
    
}
