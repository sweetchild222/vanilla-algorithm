#include <stdio.h>
#include <stdlib.h>
 
int fibo(int n) ;
 
int main(void){
	int a = 1;
	int b = 1;
	int c;

	while(c < 34){
		c = a + b;
		printf("%d\n" , c);
		a = b;
		b = c;
	}

	return 0;
									 
}
 
