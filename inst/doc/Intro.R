## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, eval = TRUE)
options(rmarkdown.html_vignette.check_title = FALSE)

## ----eval=FALSE---------------------------------------------------------------
# install.packages("BayesSUR")

## ----eval=FALSE---------------------------------------------------------------
# # library("devtools")
# devtools::install_github("zhizuio/moewishart")

## ----results='hide'-----------------------------------------------------------
library(moewishart)

n <- 200 # number of subjects
p <- 2 # dimension of covariance matrix
set.seed(123) # fix coefficients of underlying MoE model
Xq <- 3
K <- 3
betas <- matrix(runif(Xq * K, -2, 2), nrow = Xq, ncol = K)
betas[, K] <- 0

# simulate data
dat <- simData(n, p,
  Xq = 3, K = 3, betas = betas,
  pis = c(0.35, 0.40, 0.25),
  nus = c(8, 12, 3)
)

# fit Bayesian MoE-Wishart model
set.seed(123)
fit <- moewishart(
  dat$S,
  X = cbind(1, dat$X), K = 3,
  mh_sigma = c(0.2, 0.1, 0.2), # RW-MH variances (length K)
  mh_beta = c(0.3, 0.3), # RW-MH variances (length K-1)
  niter = 3000, burnin = 1000
)

## -----------------------------------------------------------------------------
burnin <- 1000
nu_mcmc <- fit$nu[-c(1:burnin), ]
colMeans(nu_mcmc)

## -----------------------------------------------------------------------------
dat$nu # true nu

## -----------------------------------------------------------------------------
MoE_Sigma <- Reduce("+", fit$Sigma) / length(fit$Sigma)
MoE_Sigma

## -----------------------------------------------------------------------------
beta_mcmc <- fit$Beta_samples[-c(1:burnin), , ]
apply(beta_mcmc, c(2, 3), mean)

## ----results='hide'-----------------------------------------------------------
# fit Bayesian Wishart mixture model
set.seed(123)
fit2 <- mixturewishart(
  dat$S,
  K = 3,
  mh_sigma = c(0.2, 0.1, 0.2), # RW-MH variances
  niter = 3000, burnin = 1000
)

## -----------------------------------------------------------------------------
colMeans(fit2$pi[-c(1:burnin), ])

## -----------------------------------------------------------------------------
colMeans(fit2$nu[-c(1:burnin), ])

## -----------------------------------------------------------------------------
# fit MoE-Wishart model via EM alg.
set.seed(123)
fit3 <- moewishart(
  dat$S,
  X = cbind(1, dat$X), K = 3,
  method = "em",
  niter = 3000
)

## -----------------------------------------------------------------------------
fit3$nu

## -----------------------------------------------------------------------------
fit3$Sigma

## -----------------------------------------------------------------------------
fit3$Beta

## -----------------------------------------------------------------------------
# fit Wishart mixture model via EM alg.
set.seed(123)
fit4 <- mixturewishart(
  dat$S,
  K = 3,
  method = "em",
  niter = 3000
)

## -----------------------------------------------------------------------------
fit4$nu

## -----------------------------------------------------------------------------
fit4$Sigma

