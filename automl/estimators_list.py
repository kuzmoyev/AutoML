from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings("ignore")

print("========= CLASSIFIERS ============")
for name, class_ in all_estimators(type_filter='classifier'):
    print(name, ',', sep='')
print()

print("========= REGRESSORS ============")
for name, class_ in all_estimators(type_filter='regressor'):
    print(name, ',', sep='')
