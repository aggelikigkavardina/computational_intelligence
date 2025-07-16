//Alexandros Kokkinos 4084, Euaggelos Tempelopoulos 4175, Aggeliki Gkavardina 4042
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define C1 1
#define C2 2
#define C3 3
#define C4 4

int findCategory(float x, float y);
void populate(FILE *file, int iterations, double x_min, double x_max,double y_min,double y_max);

int main() {
  FILE *file1, *file2;

  file1 = fopen("training_set", "w+");
  if(file1==NULL){
    perror("Opening Training Set: FAILED");
    return EXIT_FAILURE;
  }
  file2 = fopen("test_set", "w+");
  if(file2==NULL){
    perror("Opening Test Set: FAILED");
    return EXIT_FAILURE;
  }
  
  populate(file1, 4000, -1, 1, -1, 1);
  populate(file2, 4000, -1, 1, -1, 1);

  fclose(file1);
  fclose(file2);

  return 0;
}

int findCategory(float x, float y) {
  if((pow((x-0.5),2) + pow((y-0.5),2) < 0.2) && (y > 0.5)) {
    return C1;
  }
  else if((pow((x-0.5),2) + pow((y-0.5),2) < 0.2) && (y < 0.5)) {
    return C2;
  }
  else if((pow((x+0.5),2) + pow((y+0.5),2) < 0.2) && (y > -0.5)) {
    return C1;
  }
  else if((pow((x+0.5),2) + pow((y+0.5),2) < 0.2) && (y < -0.5)) {
    return C2;
  }
  else if((pow((x-0.5),2) + pow((y+0.5),2) < 0.2) && (y > -0.5)) {
    return C1;
  }
  else if((pow((x-0.5),2) + pow((y+0.5),2) < 0.2) && (y < -0.5)) {
    return C2;
  }
  else if((pow((x+0.5),2) + pow((y-0.5),2) < 0.2) && (y > 0.5)) {
    return C1;
  }
  else if((pow((x+0.5),2) + pow((y-0.5),2) < 0.2) && (y < 0.5)) {
    return C2;
  }
  else if (x*y > 0){
    return C3;
  }
  else {
    return C4;
  }
}

void populate(FILE *file, int iterations, double x_min, double x_max,double y_min,double y_max){
  int i, category;
  for(i=0; i<iterations; i++){
    double x = x_min + (rand() / (double)(RAND_MAX / (x_max - x_min)));
    double y = y_min + (rand() / (double)(RAND_MAX / (y_max - y_min)));
    category = findCategory(x, y);
    fprintf(file, "%f %f C%d\n", x , y, category);
  }
}