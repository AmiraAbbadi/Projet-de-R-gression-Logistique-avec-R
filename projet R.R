# Chargement des packages :
library(tidyverse)
library(caret)
library(dplyr)

#Étape 1:  Charger les données 

hr_Data<- HR_Analytics

#Étape 2: Préparation des données
hr_Data <- hr_Data %>%
  mutate(Attrition = factor(V4, levels = c('No', 'Yes')),
         Age = factor(V2),
         Department = factor(V7),
         Education = factor(V9),
         EducationField = factor(V10),
         Gender = factor(V14),
         MaritalStatus = factor(V20),
         PerformanceRating = factor(V28),
         OverTime = factor(V26),
         SalarySlab = factor(V22),
         PercentSalaryHike = factor(V27),
         MonthlyRate = factor(V23),
         JobRole = factor(V18),
         JobSatisfaction = factor(V19),
         EnvironmentSatisfaction = factor(V13),
         RelationshipSatisfaction =  factor(V29),
         WorkLifeBalance =  factor(V34),
         DistanceFromHome = factor(V8),
         BusinessTravel = factor(V5)
         ) %>%
  drop_na()

# Étape 3 : Exploration des Données
summary(hr_Data)
# Etape 4 Visualisation :
# Visualisation de l'attrition par département
ggplot(data = hr_Data, aes(x = Department, fill = Attrition)) + 
  geom_bar(position = "dodge", alpha = 0.8) + 
  labs(title = "Attrition by Department", x = "Department", y = "Count") +
  theme_minimal()

# Visualisation de l'attrition par Gender
ggplot(data = hr_Data, aes(x = Gender, fill = Attrition)) + 
  geom_bar(position = "dodge", alpha = 0.8) + 
  labs(title = "Attrition by Gender", x = "Gender", y = "Count") +
  theme_minimal()

# Visualisation de l'attrition par SalarySlab
ggplot(data = hr_Data, aes(x = SalarySlab, fill = Attrition)) + 
  geom_bar(position = "dodge", alpha = 0.8) + 
  labs(title = "Attrition by SalarySlab", x = "SalarySlab", y = "Count") +
  theme_minimal()

# Visualisation de l'attrition par PercentSalaryHike
ggplot(data = hr_Data, aes(x = PercentSalaryHike, fill = Attrition)) + 
  geom_bar(position = "dodge", alpha = 0.8) + 
  labs(title = "Attrition by PercentSalaryHike", x = "PercentSalaryHike", 
       y = "Count") + theme_minimal()

# Visualisation de l'attrition par JobRole
ggplot(data = hr_Data, aes(x = JobRole, fill = Attrition)) + 
  geom_bar(position = "dodge", alpha = 0.8) + 
  labs(title = "Attrition by JobRole", x = "JobRole", y = "Count") +
  theme_minimal()

# Visualisation de l'attrition par JobSatisfaction
ggplot(data = hr_Data, aes(x = JobSatisfaction, fill = Attrition)) + 
  geom_bar(position = "dodge", alpha = 0.8) + 
  labs(title = "JobSatisfaction", x = "JobSatisfaction", y = "Count") +
  theme_minimal()

# Visualisation de l'attrition par EnvironmentSatisfaction
ggplot(data = hr_Data, aes(x = EnvironmentSatisfaction, fill = Attrition)) + 
  geom_bar(position = "dodge", alpha = 0.8) + 
  labs(title = "EnvironmentSatisfaction", x = "EnvironmentSatisfaction",
       y = "Count") + theme_minimal()

# Visualisation de l'attrition par BusinessTravel
ggplot(data = hr_Data, aes(x = BusinessTravel, fill = Attrition)) + 
  geom_bar(position = "dodge", alpha = 0.8) + 
  labs(title = "BusinessTravel", x = "BusinessTravel", y = "Count") +
  theme_minimal()

# Visualisation de l'attrition par WorkLifeBalance
ggplot(data = hr_Data, aes(x = WorkLifeBalance, fill = Attrition)) + 
  geom_bar(position = "dodge", alpha = 0.8) + 
  labs(title = "WorkLifeBalance", x = "WorkLifeBalance", y = "Count") +
  theme_minimal()

# Visualisation de l'attrition par DistanceFromHome
ggplot(data = hr_Data, aes(x = DistanceFromHome, fill = Attrition)) + 
  geom_bar(position = "dodge", alpha = 0.8) + 
  labs(title = "DistanceFromHome", x = "DistanceFromHome", y = "Count") +
  theme_minimal()

# Etape 5 : Diviser les données en ensembles d'entraînement et de test
set.seed(123)
train_index <- createDataPartition(hr_Data$Attrition, p = 0.7, list = FALSE)
train_data <- hr_Data[train_index, ]
test_data <- hr_Data[-train_index, ]

# Etape 6 : Créer le modèle de régression logistique
model <- glm(Attrition ~ Department + Education + EducationField + 
               Gender + MaritalStatus + PerformanceRating + OverTime + 
               SalarySlab + PercentSalaryHike + JobSatisfaction + 
               JobRole + EnvironmentSatisfaction + RelationshipSatisfaction + 
               WorkLifeBalance + DistanceFromHome + BusinessTravel,
             data = train_data, family = binomial)

# Résumé du modèle
summary(model)
#tracer une visualisation simple des probabilités prédites d'attrition
#en fonction de la satisfaction au travail :

# Prédire les probabilités d'attrition pour les données de test
test_data$predicted_prob <- predict(model, newdata = test_data, 
                                    type = "response")

# Tracer les probabilités prédites d'attrition par niveau de satisfaction au travail
library(ggplot2)
ggplot(test_data, aes(x = factor(JobSatisfaction), y = predicted_prob)) +
  geom_boxplot() +
  labs(title = "Probabilités prédites d'attrition par niveau de satisfaction au travail",
       x = "Niveau de satisfaction au travail",
       y = "Probabilité prédite d'attrition") +
  theme_minimal()





# Étape 7 : Validation et Évaluation du Modèle

# Prédictions

# Prédictions sur les données de test
predicted <- predict(model, newdata = test_data, type = "response")
predicted 
# Créer predicted_class
predicted_class <- ifelse(predicted > 0.5, "Yes", "No")
predicted_class
# Comparer avec les valeurs réelles
actual <- as.character(test_data$Attrition)
actual

# Créer une confusion matrix
confusion_matrix <- confusionMatrix(data = factor(predicted_class, 
                                                  levels = c("Yes", "No")),                                   reference = factor(actual, levels = c("Yes", "No")))
confusion_matrix 


#Étape 8 : Tracer la courbe ROC
library(pROC)

# Prédictions sur l'ensemble de test
predicted <- predict(model, newdata = test_data, type = "response")

# Tracer la courbe ROC
roc_curve <- roc(test_data$Attrition, predicted)
plot(roc_curve, main = "Receiver Operating Characteristic (ROC) Curve", 
     col = "blue")

# Ajouter la ligne de référence (chance)
abline(a=0, b=1, col="red", lty=2)


# Étape 9 : Amélioration du Modèle
# Sélection des variables significatives
model2 <- step(model)

summary(model2)

# Réévaluation du modèle ajusté
predictions2 <- predict(model2, testData, type = 'response')
predicted_classes2 <- ifelse(predictions2 > 0.5, 'Yes', 'No')
confusionMatrix(factor(predicted_classes2), testData$Attrition)

# Interprétation des Résultats
exp(coef(model2)) # Odds Ratios
# Tracé des Rapports de Cotes des Variables Explicatives
# Calcul des rapports de cotes
odds_ratios <- exp(coef(model2))

# Supprimer l'intercept car il n'est pas pertinent pour ce graphique
odds_ratios <- odds_ratios[-1]

# Étape 10 : Tracer le graphique des rapports de cotes
barplot(odds_ratios, las=2, col="skyblue", main="Odds Ratios des Variables 
        Explicatives", ylab="Odds Ratio", xlab="Variable")

