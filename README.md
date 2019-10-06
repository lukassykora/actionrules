# Action Rules
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Action Rules (actionrules) is an implementation of Action Rules from Classification Rules algorithm described in

```Dardzinska, A. (2013). Action rules mining. Berlin: Springer.```


## Example

```python
from control import Control

control = Control()
control.read_csv("data/titanic.csv", sep="\t" ,lineterminator='\r')
control.fit(stable_antecedents = ["Age"], flexible_antecedents = ["Embarked", "Fare", "Pclass"], consequent = "Survived", conf=55, supp=3, desired_classes = ["1.0"], is_nan=False)
control.get_action_rules()
```

The output is a list of action rules and their support (support before, support after, action rule support) and confidence (confidence before, confidence after, action rule confidence).
