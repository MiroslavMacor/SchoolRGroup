# Hello, world!
#
# This is an example function named 'hello'
# which prints 'Hello, world!'.
#
# You can learn more about package authoring with RStudio at:
#
#   http://r-pkgs.had.co.nz/
#
# Some useful keyboard shortcuts for package authoring:
#
#   Build and Reload Package:  'Ctrl + Shift + B'
#   Check Package:             'Ctrl + Shift + E'
#   Test Package:              'Ctrl + Shift + T'

hello <- function() {
  print("Hello, world And have a good day !")
}

printHello <- function() {

  hello()
}

# predictor variables
X <- matrix(c(
  0,0,1,
  0,1,1,
  1,0,1,
  1,1,1
),
ncol = 3,
byrow = TRUE
)

A <- matrix(c(
  0,0,1,
  0,1,1,
  1,0,1,
  1,0,1,
  1,1,1,
  1,1,0,
  0,1,1
),

#A <- matrix(c(
#  0,0,1,
#  0,1,1,
#  1,0,1,
#  1,0,1,
#  1,1,1,
#  1,1,0,
#  0,1,1
#),

ncol = 3,
byrow = TRUE
)

# observed outcomes
y <- c(0, 1, 1, 0)


B <- c(0, 0, 1, 1,1,1,0)
# B <- c(0, 1, 1, 0,1,1,0)

# generate a random value between 0 and 1 for each
# element in X.  This will be used as our initial weights
# for layer 1
rand_vector <- runif(ncol(X) * nrow(X), -0.5, 0.5)

rand_vector_A <- runif(ncol(A) * nrow(A))

# convert above vector into a matrix
rand_matrix <- matrix(
  rand_vector,
  nrow = ncol(X),
  ncol = nrow(X),
  byrow = TRUE
)

rand_matrix_A <- matrix(
  rand_vector_A,
  nrow = ncol(A),
  ncol = nrow(A),
  byrow = TRUE
)

# this list stores the state of our neural net as it is trained
my_nn <- list(
  # predictor variables
  input = X,
  # weights for layer 1
  weights1 = rand_matrix,
  # weights for layer 2
  weights2 = matrix(runif(4), ncol = 1),
  # actual observed
  y = y,
  # stores the predicted outcome
  output = matrix(
    rep(0, times = 4),
    ncol = 1
  )
)

# this list stores the state of our neural net as it is trained
my_nn_A <- list(
  # predictor variables
  input = A,
  # weights for layer 1
  weights1 = rand_matrix_A,
  # weights for layer 2
  weights2 = matrix(runif(length(B)), ncol = 1),
  # actual observed
  y = B,
  # stores the predicted outcome
  output = matrix(
    rep(0, times = length(B)),
    ncol = 1
  )
)


# function to reate initial weights
getIniMatrix <- function(ni, numberOfAttr){

  rand_vector <- runif(ni * numberOfAttr, -0.5, 0.5)

  rand_matrix <- matrix(
    rand_vector,
    nrow = numberOfAttr,
    ncol = ni,
    byrow = TRUE
  )
  rand_matrix
}

createNetworkStructure <- function(ni, nh, no,numberOfAttr){
  my_nn_H <- list(
    # predictor variables
  input = A,

    inputOne = matrix(,
                   ncol = ni,
                   byrow = TRUE
    ),

     hiddenlayers <- list(),
     for (i in 1:nh  ) {
       if (i == 1){
         tmp <- matrix(getIniMatrix(ni,numberOfAttr),nrow = ni, ncol = ni)
         hiddenlayers[[i]] <- tmp[-(3+1):-ni,]
       }else if (i == nh){
         tmp <- matrix(getIniMatrix(ni,numberOfAttr),nrow = ni, ncol = ni)
         hiddenlayers[[i]] <- tmp[,-2:-ni]
       } else {
         hiddenlayers[[i]] <- matrix(getIniMatrix(ni,numberOfAttr),nrow = ni, ncol = ni)
       }


      },
     hiddenlayers = hiddenlayers,


    outputs <- list(),
    for (i in 1:no) {
      outputs[[i]] <- matrix(rep(0, times = ni),nrow = ni, ncol = 1)
    },
    outputs = outputs,


    # weights for layer 1
    #weights1 = hiddenlayers[[1]],
  #my_nn_H = createNetworkStructure(7,2,1,3)

  #first layer must be changed to work with input
  weights1 = hiddenlayers[[1]],


    # weights for layer 2
    weights2 = hiddenlayers[[2]],

    # actual observed
    y = B,
    # stores the predicted outcome
    output = outputs[[1]]
  )

  my_nn_H
}




#' the activation function
sigmoid <- function(x) {
  1.0 / (1.0 + exp(-x))
}

#' the derivative of the activation function
sigmoid_derivative <- function(x) {
  x * (1.0 - x)
}

loss_function <- function(nn) {
  sum((nn$y - nn$output) ^ 2)
}

feedforward <- function(nn) {

  nn$layers = nn$hiddenlayers


  nn$layers[[1]] <- sigmoid(nn$input %*% nn$hiddenlayers[[1]])


  # nn$layer2 <- sigmoid(nn$layer2 %*% nn$layer2)
  # nn$layer3 <- sigmoid(nn$layer2 %*% nn$layer3)
  # nn$layer4 <- sigmoid(nn$layer3 %*% nn$layer4)
  # nn$layer5 <- sigmoid(nn$layer4 %*% nn$layer5)
  # nn$output <- sigmoid(nn$layer5 %*% nn$weights6)

  nn$layers[[1]] <- sigmoid(nn$input %*% nn$hiddenlayers[[1]])

  layersLength <- length(nn$hiddenlayers)

  tmp <- nn$layer1
  for (i in 2:(length(nn$hiddenlayers)-1)) {
    nn$layers[[i]] <- sigmoid(nn$layers[[i-1]] %*% nn$layers[[i]])

    # This is probably supposed to be layers not hidden layers.
    #nn$layers[[1]] <- sigmoid(nn$hiddenlayers[[i-1]] %*% nn$hiddenlayers[[i]])

  }
  nn$output <- sigmoid(nn$layers[[layersLength]] %*% nn$layers[[layersLength-1]])

  # nn$layer2 <- sigmoid(nn$weights3 %*% nn$layer1)





  # nn$layer1 <- sigmoid(nn$input %*% nn$weights1)
  nn$layer1 <- sigmoid(nn$input %*% nn$hiddenlayers[[1]])

  # nn$layer2 <- sigmoid(nn$weights3 %*% nn$layer1)

 ### nn$output <- sigmoid(nn$layer1 %*% nn$hiddenlayers[[2]])
  # nn$output <- sigmoid(nn$layer1 %*% nn$weights2)



  nn
}

backprop <- function(nn) {

  # application of the chain rule to find derivative of the loss function with
  # respect to weights2 and weights1
  layersLength <- length(nn$hiddenlayers)
  #tmpWeights <- array(,dim = c(7,7,7))

  tmpWeights <- list()
  for (i in 1:7  ) {

  tmpWeights[[i]] <- matrix(getIniMatrix(7,1),nrow = 7, ncol = 7)
  }

  i<- 3
  #for (i in (layersLength):2) {
    # tmpWeights[,,i] <- (
    #   t(nn$layers[[i-1]]) %*%
    #     (2 * (nn$y - nn$output) *
    #        sigmoid_derivative(nn$output))
    # )
    #tmpWeights[[i-1]] <- ( 2 * (nn$y - nn$output) * sigmoid_derivative(nn$output)) %*% t(nn$hiddenlayers[[i]])
  if (i == 5) {
    tmpWeights[[i-1]] <- ( 2 * (nn$y - nn$output) * sigmoid_derivative(nn$output)) %*% (nn$hiddenlayers[[5]])
  }else {
    tmpWeights[[i-1]] <- ( 2 * (nn$y - nn$output) * sigmoid_derivative(nn$output)) %*% t(nn$hiddenlayers[[i]])
  }


    tmpWeights[[i-1]] <- tmpWeights[[i-1]] * sigmoid_derivative(nn$layers[[i-1]])
    tmpWeights[[i-1]] <- t(nn$input) %*% tmpWeights[[i-1]]

    #nn$hiddenlayers[[i-1]] <- nn$hiddenlayers[[i-1]] + tmpWeights[[i-1]]
    #nn$hiddenlayers[[i]] <- nn$hiddenlayers[[i]] + tmpWeights[[i]]

  #}


  # d_weights2 <- (
  #   t(nn$layer1) %*%
  #     # `2 * (nn$y - nn$output)` is the derivative of the sigmoid loss function
  #     (2 * (nn$y - nn$output) *
  #        sigmoid_derivative(nn$output))
  # )

  # d_weights1 <- ( 2 * (nn$y - nn$output) * sigmoid_derivative(nn$output)) %*%
  #   t(nn$weights2)
  # d_weights1 <- d_weights1 * sigmoid_derivative(nn$layer1)
  # d_weights1 <- t(nn$input) %*% d_weights1
  #
  # # update the weights using the derivative (slope) of the loss function
  # nn$weights1 <- nn$weights1 + d_weights1
  # nn$weights2 <- nn$weights2 + d_weights2

  nn
}

# number of times to perform feedforward and backpropagation
n <- 1500

# data frame to store the results of the loss function.
# this data frame is used to produce the plot in the
# next code chunk
loss_df <- data.frame(
  iteration = 1:n,
  loss = vector("numeric", length = n)
)

# for (i in seq_len(1500)) {
# #   my_nn <- feedforward(my_nn)
# #   my_nn <- backprop(my_nn)
# #
# #   # store the result of the loss function.  We will plot this later
# #   loss_df$loss[i] <- loss_function(my_nn)
# # }
# #
# # for (i in seq_len(1500)) {
# #   my_nn_A <- feedforward(my_nn_A)
# #   my_nn_A <- backprop(my_nn_A)
# #
# #   # store the result of the loss function.  We will plot this later
# #   loss_df$loss[i] <- loss_function(my_nn_A)
# # }

my_nn_H = createNetworkStructure(7,5,1,3)


for (i in seq_len(1500)) {
  my_nn_H <- feedforward(my_nn_H)
  my_nn_H <- backprop(my_nn_H)

  # store the result of the loss function.  We will plot this later
  loss_df$loss[i] <- loss_function(my_nn_H)
}


# print the predicted outcome next to the actual outcome
data.frame(
  "Predicted" = round(my_nn$output, 3),
  "Actual" = y
)
##   Predicted Actual
## 1     0.017      0
## 2     0.975      1
## 3     0.982      1
## 4     0.024      0























