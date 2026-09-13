# Instructieset voor Claude AI -- Jupyter Notebook (Python) — Pragmatisch DRY & Clean Code

## 1) DRY-principe -- harde eisen

Pas DRY toe door:

-   Geen dubbele code voor plotting, evaluatie of data-transformaties.
-   Herbruikbare functies voor herhaalde taken (bijv. plot-helpers, model-evaluatie).
-   Centraliseer configuratie in een `Config` dataclass:
    -   feature-lijsten (numeriek, categorisch)
    -   paden (dataset, model)
    -   constanten (random state, test size, cv folds)
-   One source of truth voor feature-definities en modelconfiguratie.

------------------------------------------------------------------------

## 2) SOLID-principes -- pragmatische toepassing

### S --- Single Responsibility

-   Functies doen één ding (laden, plotten, evalueren, serialiseren).
-   Pipeline-stappen zijn losse componenten via sklearn `Pipeline` + `ColumnTransformer`.

### O --- Open/Closed

-   Nieuwe modellen toevoegen door een nieuwe pipeline te maken, zonder bestaande code te wijzigen.
-   sklearn's Pipeline/ColumnTransformer biedt deze abstractie al.

### L --- Liskov Substitution

-   Alle sklearn-modellen zijn inwisselbaar binnen dezelfde pipeline-structuur.
-   Geen custom abstracte classes nodig — sklearn's API is het contract.

### I --- Interface Segregation

-   Kleine, gerichte functies in plaats van grote god-functies.
-   Gescheiden verantwoordelijkheden: data loading, EDA, preprocessing, training, evaluatie.

### D --- Dependency Inversion

-   De pipeline hangt af van sklearn-abstracties (Estimator, Transformer), niet van concrete implementaties.
-   Config dataclass wordt doorgegeven aan functies (geen hardcoded waarden).

------------------------------------------------------------------------

## 3) Code-kwaliteitsregels

-   Gebruik `dataclasses` voor de centrale Config.
-   Geen abstracte base classes of factory patterns (overkill voor 2 modellen).
-   Geen custom exceptions (sklearn errors volstaan).
-   Geen god objects of god functions.
-   Korte docstring (1-2 regels) bij elke functie.

------------------------------------------------------------------------

## 4) Stijlrichtlijnen

-   Markdown: compact en helder.
-   Code leidend, markdown ondersteunend.
-   Engelse code comments, consistent door het hele notebook.
-   Elke code cel begint met `# === PHASE XX: NAME ===` voor navigatie.

------------------------------------------------------------------------

## 5) Output format

Lever het notebook als opeenvolgende Markdown- en Code-cellen, zodat het
direct te gebruiken is in Google Colab.
