//Alexandros Kokkinos 4084, Euaggelos Tempelopoulos 4175, Aggeliki Gkavardina 4042
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define D 2
#define K 4
#define H1 64
#define H2 32
#define H3 16
#define TANH 1
#define RELU 2
#define FUNC TANH
#define LEARNING_RATE 0.05
#define THRESHOLD 0.001
#define N 4000
#define BATCH_SIZE (N/200)

/*##################################### GLOBALS #########################################*/

double category_matrix[N][6] = {0}; // x1, x2, t1, t2, t3, t4

double w1[H1][D];
double w2[H2][H1];
double w3[H3][H2];
double w4[K][H3];
double b1[H1];
double b2[H2];
double b3[H3];
double b4[K];

double changes_layer1[H1][D];
double changes_layer2[H2][H1];
double changes_layer3[H3][H2];
double changes_layer4[K][H3];
double changes_b1[H1];
double changes_b2[H2];
double changes_b3[H3];
double changes_b4[K];

double h1[N][H1];
double h2[N][H2];
double h3[N][H3];
double output[N][K];
double delta_h1[N][H1];
double delta_h2[N][H2];
double delta_h3[N][H3];
double delta_out[N][K];

double previous_err = INFINITY;

/*##################################### PROTOS #########################################*/

void load_file(FILE *file);
void initialize_weights();

double activation_function(double x, int isOutput);
double gradient_activation_function(double x, int isOutput);
void forward_pass(double *x, int idx);
void backprop(double *x, int idx);

void update_weights();
void clear_accs();

double calc_err();
double eval_accu();
void gradient_descent();
void check_results();

/*##################################### MAIN #########################################*/

int main() {
  FILE *file1, *file2;

  file1 = fopen("training_set", "r");
  if (file1 == NULL) {
    perror("Opening training_set: FAILED");
    return EXIT_FAILURE;
  }

  file2 = fopen("test_set", "r");
  if (file2 == NULL) {
    perror("Opening test_set: FAILED");
    fclose(file1);
    return EXIT_FAILURE;
  }

  load_file(file1);
  gradient_descent();
  check_results();

  fclose(file1);
  fclose(file2);
  return 0;
}

/*##################################### FUNCTIONS #########################################*/

void load_file(FILE *file) { //load values x and y plus the category next to them.
  int counter = 0, category;
  char *line = NULL;
  size_t len = 0;

  while (counter < N && getline(&line, &len, file) != -1) {
    int ret = sscanf(line, "%lf %lf C%d", &category_matrix[counter][0], &category_matrix[counter][1], &category);
    if (ret != 3) {
      fprintf(stderr, "Error parsing line %d: %s\n", counter, line);
      free(line);
      exit(EXIT_FAILURE);
    }
    if (category >= 1 && category <= 4) {
      category_matrix[counter][2 + category - 1] = 1.0;
    }
    counter++;
  }
  free(line);

  if (counter < N) {
    fprintf(stderr, "Warning: Only %d rows loaded from training set.\n", counter);
  }
  if (counter > N) {
    fprintf(stderr, "Error: Training file contains more than %d rows.\n", N);
    exit(EXIT_FAILURE);
  }
}


void initialize_weights() { //randomly initialise weights [-1,1]

  int i, j;

  for(i=0; i<H1; i++) {
    for(j=0; j<D; j++){
     w1[i][j] = (-1) + (rand() / (double)(RAND_MAX / (double)((1) - (-1)))); 
    }
  }

  for(i=0; i<H2; i++) {
    for(j=0; j<H1; j++){
      w2[i][j] = (-1) + (rand() / (double)(RAND_MAX / (double)((1) - (-1))));
    }
  }

  for(i=0; i<H3; i++) {
    for(j=0; j<H2; j++) {
      w3[i][j] = (-1) + (rand() / (double)(RAND_MAX / (double)((1) - (-1))));
    }
  }

  for(i=0; i<K; i++) {
    for(j=0; j<H3; j++) {
      w4[i][j] = (-1) + (rand() / (double)(RAND_MAX / (double)((1) - (-1))));
    }
  }

  for(i=0; i<H1; i++) {
    b1[i] = (-1) + (rand() / (double)(RAND_MAX / (double)((1) - (-1))));
  }

  for(i=0; i<H2; i++) {
    b2[i] = (-1) + (rand() / (double)(RAND_MAX / (double)((1) - (-1))));
  }

  for(i=0; i<H3; i++) {
    b3[i] = (-1) + (rand() / (double)(RAND_MAX / (double)((1) - (-1))));
  }

  for(i=0; i<K; i++) {
    b4[i] = (-1) + (rand() / (double)(RAND_MAX / (double)((1) - (-1))));
  }
}

double activation_function(double x, int isOutput) { //tanh and relu funcs
  if(FUNC == TANH || isOutput==0) {
    return (exp(x) - exp(-x)) / (double)(exp(x) + exp(-x));
  }
  else if(FUNC == RELU && isOutput==1) {
    return (x + fabs(x))/2;
  } 
  else {
        fprintf(stderr, "Invalid activation function configuration (FUNC).\n");
        exit(EXIT_FAILURE);
    }
}

double gradient_activation_function(double x, int isOutput) { // the derivatives of the functions above
    if (FUNC == TANH || isOutput == 0) {
        return 1 - x * x;
    } else if (FUNC == RELU && isOutput==1) {
        return x > 0.0 ? 1.0 : 0.0;
    } 
    else {
        fprintf(stderr, "Invalid activation function configuration (FUNC).\n");
        exit(EXIT_FAILURE);
    }
}

void forward_pass(double *x, int idx) {
  int i, j;
  double sum;

  for (i = 0; i < H1; i++) {
    sum = b1[i];
    for (j = 0; j < D; j++) {
      sum += w1[i][j] * x[j];
    }
    h1[idx][i] = activation_function(sum, 0); //goes to hidden layer 1
  }

  for (i = 0; i < H2; i++) {
    sum = b2[i];
    for (j = 0; j < H1; j++) {
      sum += w2[i][j] * h1[idx][j];
    }
    h2[idx][i] = activation_function(sum, 0); //goes to hidden layer 2
  }

  for (i = 0; i < H3; i++) {
    sum = b3[i];
    for (j = 0; j < H2; j++) {
      sum += w3[i][j] * h2[idx][j];
    }
    h3[idx][i] = activation_function(sum, 0); //goes to hidden layer 3
  }

  for (i = 0; i < K; i++) {
    sum = b4[i];
    for (j = 0; j < H3; j++) {
      sum += w4[i][j] * h3[idx][j];
    }
    output[idx][i] = activation_function(sum, 1); //goes to output
  }
}

void backprop(double *x, int idx) {
  int i, j;
  double sum;

  //for output weights
  for (i = 0; i < K; i++) {
    delta_out[idx][i] = (output[idx][i] - category_matrix[idx][2 + i]) * gradient_activation_function(output[idx][i], 1);
  }

  //for layer 3 weights
  for (i = 0; i < H3; i++) {
    sum = 0;
    for (j = 0; j < K; j++) {
      sum += delta_out[idx][j] * w4[j][i];
    }
    delta_h3[idx][i] = sum * gradient_activation_function(h3[idx][i], 0);
  }

  //for layer 2 weights
  for (i = 0; i < H2; i++) {
    sum = 0;
    for (j = 0; j < H3; j++) {
      sum += delta_h3[idx][j] * w3[j][i];
    }
    delta_h2[idx][i] = sum * gradient_activation_function(h2[idx][i], 0);
  }

  //for layer 1 weights
  for (i = 0; i < H1; i++) {
    sum = 0;
    for (j = 0; j < H2; j++) {
      sum += delta_h2[idx][j] * w2[j][i];
    }
    delta_h1[idx][i] = sum * gradient_activation_function(h1[idx][i], 0);
  }

  //for layer 1 bias
  for (i = 0; i < H1; i++) {
    for (j = 0; j < D; j++) {
      changes_layer1[i][j] += delta_h1[idx][i] * x[j];
    }
    changes_b1[i] += delta_h1[idx][i];
  }

  //for layer 2 bias
  for (i = 0; i < H2; i++) {
    for (j = 0; j < H1; j++) {
      changes_layer2[i][j] += delta_h2[idx][i] * h1[idx][j];
    }
    changes_b2[i] += delta_h2[idx][i];
  }

  //for layer 3 bias
  for (i = 0; i < H3; i++) {
    for (j = 0; j < H2; j++) {
      changes_layer3[i][j] += delta_h3[idx][i] * h2[idx][j];
    }
    changes_b3[i] += delta_h3[idx][i];
  }

  //for output bias
  for (i = 0; i < K; i++) {
    for (j = 0; j < H3; j++) {
      changes_layer4[i][j] += delta_out[idx][i] * h3[idx][j];
    }
    changes_b4[i] += delta_out[idx][i];
  }
}


void update_weights() { //change weights based on learning rate and batch size
  int i, j;

  for (i = 0; i < H1; i++) {
    for (j = 0; j < D; j++) {
      w1[i][j] -= LEARNING_RATE * changes_layer1[i][j] / BATCH_SIZE;
    }
    b1[i] -= LEARNING_RATE * changes_b1[i] / BATCH_SIZE;
  }

  for (i = 0; i < H2; i++) {
    for (j = 0; j < H1; j++) {
      w2[i][j] -= LEARNING_RATE * changes_layer2[i][j] / BATCH_SIZE;
    }
    b2[i] -= LEARNING_RATE * changes_b2[i] / BATCH_SIZE;
  }

  for (i = 0; i < H3; i++) {
    for (j = 0; j < H2; j++) {
      w3[i][j] -= LEARNING_RATE * changes_layer3[i][j] / BATCH_SIZE;
    }
    b3[i] -= LEARNING_RATE * changes_b3[i] / BATCH_SIZE;
  }

  for (i = 0; i < K; i++) {
    for (j = 0; j < H3; j++) {
      w4[i][j] -= LEARNING_RATE * changes_layer4[i][j] / BATCH_SIZE;
    }
    b4[i] -= LEARNING_RATE * changes_b4[i] / BATCH_SIZE;
  }

  clear_accs();
}

void clear_accs() { 
  int i, j;

  for (i = 0; i < H1; i++) {
    for (j = 0; j < D; j++) {
      changes_layer1[i][j] = 0;
    }
    changes_b1[i] = 0;
  }

  for (i = 0; i < H2; i++) {
    for (j = 0; j < H1; j++) {
      changes_layer2[i][j] = 0;
    }
    changes_b2[i] = 0;
  }

  for (i = 0; i < H3; i++) {
    for (j = 0; j < H2; j++) {
      changes_layer3[i][j] = 0;
    }
    changes_b3[i] = 0;
  }

  for (i = 0; i < K; i++) {
    for (j = 0; j < H3; j++) {
      changes_layer4[i][j] = 0;
    }
    changes_b4[i] = 0;
  }
}

double calc_err() { //find all diff medians and divide by the amount of points
  int i, j;
  double error = 0, diff;

  for (i = 0; i < N; i++) {
    for (j = 0; j < K; j++) {
      diff = category_matrix[i][2 + j] - output[i][j];
      error += (diff * diff)/2;
    }
  }

  return error / (double)N;
}

double eval_accu() { //calc how many points are in the correct class
  int correct = 0, i, j, target_class, predicted_class;
  double test_accu;

  for (i = 0; i < N; i++) {
    target_class = 0;
    for (j = 1; j < K; j++) {
      if (category_matrix[i][2 + j] > category_matrix[i][2 + target_class]) {
        target_class = j;
      }
    }

    predicted_class = 0;
    for (j = 1; j < K; j++) {
      if (output[i][j] > output[i][predicted_class]) {
        predicted_class = j;
      }
    }

    if (predicted_class == target_class) {
      correct++;
    }
  }
  test_accu = (double)correct / N * 100.0;
  return test_accu;
}

void gradient_descent(){ //run for at least 800 epochs
  int epoch, i;
  double err, accu;
  initialize_weights();

  epoch = 0;
  while (1) {
    clear_accs();
    for (i = 0; i < N; i++) {
      forward_pass(category_matrix[i], i);
      backprop(category_matrix[i], i);

      if ((i + 1) % BATCH_SIZE == 0) {
        update_weights();
      }
    }

    err = calc_err();
    accu = eval_accu();
    printf("Epoch %d: Error = %.6f, Accuracy = %.2f%%\n", epoch, err, accu);

    if (epoch >= 800 && fabs(previous_err - err) < THRESHOLD) {
      break;
    }
    previous_err = err;
    epoch++;
  }
}

void check_results(){ //differentiate correct and wrong answers by adding (+) or (-) respectively
   int i;
   for (i = 0; i<N; i++){
    if(round(output[i][0])== category_matrix[i][2] &&
      round(output[i][1])== category_matrix[i][3] &&
      round(output[i][2])== category_matrix[i][4] &&
      round(output[i][3])== category_matrix[i][5]){
        printf("(+) %lf\t %lf\n",category_matrix[i][0],category_matrix[i][1]);
    }else{
      printf("(-) %lf\t %lf\n",category_matrix[i][0],category_matrix[i][1]);
    }
   }
}

