# Import queueing package
#install.packages("queueing")
#install.packages("simone")

library(queueing)

#----------------------------MM1 Model --------------------------------

# Set queue model input parameters
input_mm1 <- NewInput.MM1(lambda = 2, mu = 3, n = 100)


# Create queue class object
output_mm1 <- QueueingModel(input_mm1)

# Get queue model report
Report(output_mm1)

# Get queue model summary
summary(output_mm1)

#----------------------------- MMC Model---------------------------------


# Set queue model input parameters
input_mm2 <- NewInput.MMC(lambda = 2, mu = 3, c=2, n=100, method = 0)


# Create queue class object
output_mm2 <- QueueingModel(input_mm2)

# Get queue model report
Report(output_mm2)

# Get queue model summary
summary(output_mm2)

# Pn.o_MM1(output_mm1)
# 
# ?NewInput.MMCKM()
# 
# 
# P = Pn(QueueingModel.i_MMC(input_mm2))
# 
# plot(P,type= "l")

# Summary Output Nomenclatures
# RO (??): overall system utilization
# P0: the probability that all servers are idle
# Lq: long-run average number of customers in the queue
# Wq: long-run average time spent in the queue
# X: system throughput
# L: expected number of customers in queueing system
# W: lexpected time spent in the system
# Wqq: long-run average time spent in queue when there is queue in a queueing model
# Lqq: long-run average number of customers in queue when there is queue in a queueing model


curve(dpois(x, input_mm2$lambda),
      from = 0, 
      to = 20, 
      type = "b", 
      lwd = 2,
      xlab = "Number of customers",
      ylab = "Probability",
      main = "Poisson Distribution for Arrival Process",
      ylim = c(0, 0.30),
      n = 21)


curve(dexp(x, rate = 1/input_mm2$lambda),
      from = 0, 
      to = 10,
      type = "l", 
      lwd = 2,
      xlab = "Interarrival Time",
      ylab = "Probaility",
      main = "Exponential Distribution for Interarrival Time",
      ylim = c(0, 1))
abline(h = 0)


curve(dexp(x, rate = input_mm1$mu),
      from = 0, 
      to = 5, 
      type = "l", 
      lwd = 2,
      xlab = "Service Waiting Time",
      ylab = "Probaility",
      main = "Exponential Distribution for Service Process",
      ylim = c(0, 1))
abline(h = 0)
