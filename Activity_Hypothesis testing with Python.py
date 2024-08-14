import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


data = pd.read_csv('marketing_sales_data.csv')
data.head()
# **Question:** Why is it useful to perform exploratory data analysis before constructing a linear regression model?
#* To understand which variables are present in the data
# * To consider the distribution of features, such as minimum, mean, and maximum values
# * To plot the relationship between the independent and dependent variables and visualize which features have a linear relationship
# * To identify issues with the data, such as incorrect or missing values.


sns.boxplot(x= 'TV', y='Sales', data = data)
# **Question:** Is there variation in `Sales` based off the `TV` promotion budget?
# There is considerable variation in `Sales` across the `TV` groups. The significance of these differences can be tested with a one-way ANOVA.


data = data.dropna(axis=0)
data.isnull().sum(axis=0)



ols_formula = 'Sales ~ C(TV)'
OLS = ols(formula = ols_formula, data = data)
model = OLS.fit()
model_summary = model.summary()
model_summary

# **Question:** Which categorical variable did you choose for the model? Why?
# * `TV` was selected as the preceding analysis showed a strong relationship between the `TV` promotion budget and the average `Sales`.
# * `Influencer` was not selected because it did not show a strong relationship to `Sales` in the analysis.



# **Question:** Is the linearity assumption met?
# Because the model does not have any continuous independent variables, the linearity assumption is not required. 
# The independent observation assumption states that each observation in the dataset is independent. As each marketing promotion (row) is independent from one another, the independence assumption is not violated.
# Next, verify that the normality assumption is upheld for the model.


residuals = model.resid
fig, axes = plt.subplots(1, 2, figsize = (8,4))
sns.histplot(residuals, ax=axes[0])
axes[0].set_xlabel("Residual Value")
axes[0].set_title("Histogram of Residuals")

sm.qqplot(residuals, line='s',ax = axes[1])
axes[1].set_title("Normal QQ Plot")
plt.tight_layout()
plt.show()

# **Question:** Is the normality assumption met?
# There is reasonable concern that the normality assumption is not met when `TV` is used as the independent variable predicting `Sales`. The normal q-q forms an 'S' that deviates off the red diagonal line, which is not desired behavior. 



fig = sns.scatterplot(x = model.fittedvalues, y = model.resid)
fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")
fig.set_title("Fitted Values v. Residuals")
fig.axhline(0)
plt.show()

# **Question:** Is the constant variance (homoscedasticity) assumption met?
# The variance where there are fitted values is similarly distributed, validating that the constant variance assumption is met.

model.summary()


# **Question:** What is your interpretation of the model's R-squared?
# Using `TV` as the independent variable results in a linear regression model with $R^{2} = 0.874$. In other words, the model explains $87.4\%$ of the variation in `Sales`. This makes the model an effective predictor of `Sales`. 

# **Question:** What is your intepretation of the coefficient estimates? Are the coefficients statistically significant?
# The default `TV` category for the model is `High`, because there are coefficients for the other two `TV` categories, `Medium` and `Low`. According to the model, `Sales` with a `Medium` or `Low` `TV` category are lower on average than `Sales` with a `High` `TV` category. For example, the model predicts that a `Low` `TV` promotion would be 208.813 (in millions of dollars) lower in `Sales` on average than a `High` `TV` promotion. 
# The p-value for all coefficients is $0.000$, meaning all coefficients are statistically significant at $p=0.05$. The 95% confidence intervals for each coefficient should be reported when presenting results to stakeholders. For instance, there is a $95\%$ chance the interval $[-215.353,-202.274]$ contains the true parameter of the slope of $\beta_{TVLow}$, which is the estimated difference in promotion sales when a `Low` `TV` promotion is chosen instead of a `High` `TV` promotion.

# **Question:** Do you think your model could be improved? Why or why not? How?
# Given how accurate `TV` was as a predictor, the model could be improved with a more granular view of the `TV` promotions, such as additional categories or the actual `TV` promotion budgets. Further, additional variables, such as the location of the marketing campaign or the time of year, may increase model accuracy. 


sm.stats.anova_lm(model, type = 2)

# **Question:** What are the null and alternative hypotheses for the ANOVA test?
# The null hypothesis is that there is no difference in `Sales` based on the `TV` promotion budget.
# The alternative hypothesis is that there is a difference in `Sales` based on the `TV` promotion budget.

#**Question:** What is your conclusion from the one-way ANOVA test?
# The results of the one-way ANOVA test indicate that you can reject the null hypothesis in favor of the alternative hypothesis. There is a statistically significant difference in `Sales` among `TV` groups.

# **Question:** What did the ANOVA test tell you?
# The results of the one-way ANOVA test indicate that you can reject the null hypothesis in favor of the alternative hypothesis. There is a statistically significant difference in `Sales` among `TV` groups.


tukey_oneway = pairwise_tukeyhsd(endog = data['Sales'], groups = data['TV'])
tukey_oneway.summary()

# **Question:** What is your interpretation of the Tukey HSD test?
# The first row, which compares the `High` and `Low` `TV` groups, indicates that you can reject the null hypothesis that there is no significant difference between the `Sales` of these two groups. 
# You can also reject the null hypotheses for the two other pairwise comparisons that compare `High` to `Medium` and `Low` to `Medium`.

# **Question:** What did the post hoc tell you?**
# A post hoc test was conducted to determine which `TV` groups are different and how many are different from each other. This provides more detail than the one-way ANOVA results, which can at most determine that at least one group is different. Further, using the Tukey HSD controls for the increasing probability of incorrectly rejecting a null hypothesis from peforming multiple tests. 
# The results were that `Sales` is not the same between any pair of `TV` groups. 
