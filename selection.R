# Forward, Backward and Stepwise selection methods

# Importing the data
data = read.csv(file.choose())
data = subset(data, select=-X)

# Full logistic model
full_model = glm(status~.-1,data=data,family=binomial)
summary(full_model)
# Forward Selection
forward = step(full_model, direction = 'forward')
summary(forward)

# Backward Selection
backward = step(full_model, direction = 'backward')
summary(backward)

# Stepwise selection
stepwise = step(full_model, direction = 'both')
summary(stepwise)