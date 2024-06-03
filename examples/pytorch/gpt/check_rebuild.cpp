#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdio.h>
#include <string.h>

int main() {

    int num_layer = 24;
    int beam_width = 1;
    int batch_size = 1;
    int hidden_units = 1024;

    int hidden_units_proc = hidden_units/4;
    int prompt_size = 5;
    int seq_len = 10;
    int size_per_head = 64;

    int scale = num_layer * beam_width * batch_size;

    size_t cache_size = num_layer * 1000 * beam_width * batch_size * hidden_units * 4; // sizeof float
    size_t cache_size_proc = cache_size/4;
    
    size_t token_cache_size = cache_size/1000;
    size_t token_cache_size_proc = token_cache_size/4;

    size_t output_size = batch_size * beam_width * 1000 * 4;
    size_t total_size = 2*cache_size + output_size;
    size_t total_size_proc = 2*cache_size_proc + output_size;

    // map the 4 files
    void* cache_addrs[4];
    
    int fd0 = open("/workspace/dejavu-ft/build/test_file_0", O_RDONLY, (mode_t)0666);
    cache_addrs[0] = mmap(NULL, total_size_proc, PROT_READ, MAP_SHARED, fd0, 0);   

    int fd1 = open("/workspace/dejavu-ft/build/test_file_1", O_RDONLY, (mode_t)0666);
    cache_addrs[1] = mmap(NULL, total_size_proc, PROT_READ, MAP_SHARED, fd1, 0);

    int fd2 = open("/workspace/dejavu-ft/build/test_file_2", O_RDONLY, (mode_t)0666);
    cache_addrs[2] = mmap(NULL, total_size_proc, PROT_READ, MAP_SHARED, fd2, 0);

    int fd3 = open("/workspace/dejavu-ft/build/test_file_3", O_RDONLY, (mode_t)0666);
    cache_addrs[3] = mmap(NULL, total_size_proc, PROT_READ, MAP_SHARED, fd3, 0);

    // open file for writing
    int fd = open("/workspace/dejavu-ft/build/test_file_all", O_CREAT | O_RDWR | O_TRUNC, (mode_t)0666);    
    ftruncate(fd, total_size);
    void* host_addr = mmap(NULL, total_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    
    // write - key cache
    for (int i=0; i<prompt_size+seq_len; i++) {
	char* start_addr = (char*)host_addr + i*token_cache_size;
	for (int k=0; k<scale; k++) {
                char* start_k_addr = start_addr + k*hidden_units*4;
                for (int j=0; j<4; j++) {
                        char* src = (char*)(cache_addrs[j]) + i*token_cache_size_proc + k*hidden_units_proc*4;
                        char* dst = start_k_addr + j*size_per_head*4*4; // 4 is float size
                        memcpy(dst,src,size_per_head*4*4); // 4 is float size
                }
        }
	/*for (int j=0; j<4; j++) {
		char* src = (char*)(cache_addrs[j]) + i*token_cache_size_proc;
		memcpy(start_addr, src, token_cache_size_proc);
		start_addr += token_cache_size_proc;
	}*/
    }

    // write - value cache
    /*for (int i=0; i<prompt_size+seq_len; i++) {
	char* start_addr = (char*)host_addr + cache_size +i*token_cache_size;
        for (int j=0; j<4; j++) {
		char* src = (char*)(cache_addrs[j]) + cache_size_proc + i*token_cache_size_proc;
                memcpy(start_addr, src, token_cache_size_proc);
                start_addr += token_cache_size_proc;
        }
    }*/
    for (int i=0; i<prompt_size+seq_len; i++) {
        char* start_addr = (char*)host_addr + cache_size + i*token_cache_size;
        for (int k=0; k<scale; k++) {
                char* start_k_addr = start_addr + k*hidden_units*4;
                for (int j=0; j<4; j++) {
                        char* src = (char*)(cache_addrs[j]) + cache_size_proc + i*token_cache_size_proc + k*hidden_units_proc*4;
                        char* dst = start_k_addr + j*size_per_head*4*4; // 4 is float size
                        memcpy(dst,src,size_per_head*4*4); // 4 is float size
                }
        }
    }
	

    // write - output
    char* dst = (char*)host_addr + 2*cache_size;
    char* src = (char*)(cache_addrs[0]) + 2*cache_size_proc;
    memcpy(dst, src, output_size);

    // msync
    if (msync(host_addr, total_size, MS_SYNC)) {
          perror("msync failed with error:");
    }

}
