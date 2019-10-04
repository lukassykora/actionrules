# Action Rules
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Action Rules (actionrules) is an implementation of Action Rules from Classification Rules algorithm described in

```Dardzinska, A. (2013). Action rules mining. Berlin: Springer.```


## Example

```python
from control import Control

control = Control()
control.read_csv("data/titanic.csv", sep="\t" ,lineterminator='\r')
control.fit(desired_classes = ["1.0"], stable_antecedents = ["Age"], flexible_antecedents = ["Embarked", "Fare", "Pclass"], consequent = "Survived", conf=55, supp=3, is_nan=False)
control.get_action_rules()
```

The output is a dictionary where the key is an action rule and the value is a tuple of (support before, support after, action rule support) and (confidence before, confidence after, action rule confidence).