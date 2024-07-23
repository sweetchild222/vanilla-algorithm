#include <stdio.h>
#include <stdlib.h>
 
int fibo(int n) ;
 
int main(void){
	int a = 0;
	int b = 1;
	int c;

	do{
		c = a + b;
		printf("%d\n" , c);
		a = b;
		b = c;
	}while(c < 34);

	return 0;
									 
}
 
