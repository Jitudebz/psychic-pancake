library(simmer)
library(simmer.plot)

set.seed(1933)

env <- simmer()

Patient <-
  trajectory("Patient's path") %>%
  set_attribute("start_time", function() {now(env)}) %>%
  log_(function() {
    paste("Queue is", get_queue_count(env, "doctor1"), "on arrival")
  }) %>%
  set_attribute("start_time", function() {now(env)}) %>%
  log_(function() {
    paste("Queue is", get_queue_count(env, "doctor2"), "on arrival")
  }) %>%
  
  seize("doctor1") %>%
  log_(function() {paste("Waited", now(env) - get_attribute(env, "start_time"))}) %>%
  timeout(12) %>%
  release("doctor1") %>%
  
  seize("doctor2") %>%
  log_(function() {paste("Waited", now(env) - get_attribute(env, "start_time"))}) %>%
  timeout(12) %>%
  release("doctor2") %>%
  
  log_("Completed")

env <-
  simmer("env") %>%
  add_resource("doctor1", preemptive = F) %>%
  add_resource("doctor2", preemptive = F) %>%
  add_generator("Patient", Patient, function() {c(0, rexp(4, 1/10), -1)}) %>%
  add_generator("John", Patient, at(2), priority = 1)%>%
  add_generator("Maria",Patient, at(43), priority = 2) %>%
  add_generator("Jimmy", Patient, at(33), priority = 3)

env %>% run(until = 540)


env %>%
  get_mon_arrivals() %>%
  transform(waiting_time = end_time - start_time - activity_time)


resources <- get_mon_resources(env)
plot(resources, metric = "utilization")


plot(resources, metric = "usage", c("doctor1", "doctor2"), items = "server")



arrivals <- get_mon_arrivals(env)
plot(arrivals, metric = "flow_time")


arrivals <- get_mon_arrivals(env)
plot(arrivals, metric = "activity_time")


arrivals <- get_mon_arrivals(env)
plot(arrivals, metric = "waiting_time")

