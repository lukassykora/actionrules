"""Class Output."""

import json


class Output:
    """
    A class used to format and export action rules.

    Attributes
    ----------
    action_rules : list
        List containing the action rules.
    target : str
        The target attribute for the action rules.

    Methods
    -------
    get_ar_notation()
        Generate a string representation of the action rules in a human-readable format.
    get_export_notation()
        Generate a list of dictionaries representing the action rules for export.
    get_pretty_ar_notation()
        Generate a list of text strings representing the action rules.
    """

    def __init__(self, action_rules: list, target: str):
        """
        Initialize the Output class with the specified action rules and target attribute.

        Parameters
        ----------
        action_rules : list
            List containing the action rules.
        target : str
            The target attribute for the action rules.
        """
        self.action_rules = action_rules
        self.target = target

    def get_ar_notation(self):
        """
        Generate a string representation of the action rules in a human-readable format.

        Returns
        -------
        str
            String representation of the action rules.
        """
        ar_notation = []
        for action_rule in self.action_rules:
            rule = '['
            for i, item in enumerate(action_rule['undesired']['itemset']):
                if i > 0:
                    rule += ' ∧ '
                rule += '('
                if item == action_rule['desired']['itemset'][i]:
                    if '_<item_stable>_' in item:
                        val = item.split('_<item_stable>_')
                        rule += str(val[0]) + ': ' + str(val[1])
                    else:
                        val = item.split('_<item_flexible>_')
                        rule += str(val[0]) + '*: ' + str(val[1])
                else:
                    val = item.split('_<item_flexible>_')
                    val_desired = action_rule['desired']['itemset'][i].split('_<item_flexible>_')
                    rule += str(val[0]) + ': ' + str(val[1]) + ' → ' + str(val_desired[1])
                rule += ')'
            rule += (
                '] ⇒ ['
                + str(self.target)
                + ': '
                + str(action_rule['undesired']['target'])
                + ' → '
                + str(action_rule['desired']['target'])
                + ']'
            )
            rule += (
                ', support of undesired part: '
                + str(action_rule['undesired']['support'])
                + ', confidence of undesired part: '
                + str(action_rule['undesired']['confidence'])
            )
            rule += (
                ', support of desired part: '
                + str(action_rule['desired']['support'])
                + ', confidence of desired part: '
                + str(action_rule['desired']['confidence'])
            )
            rule += ', uplift: ' + str(action_rule['uplift'])
            ar_notation.append(rule)
        return ar_notation

    def get_export_notation(self):
        """
        Generate a list of dictionaries representing the action rules for export.

        Returns
        -------
        list
            List of dictionaries representing the action rules.
        """
        rules = []
        for ar_dict in self.action_rules:
            rule = {'stable': [], 'flexible': []}
            for i, item in enumerate(ar_dict['undesired']['itemset']):
                if item == ar_dict['desired']['itemset'][i]:
                    if '_<item_stable>_' in item:
                        val = item.split('_<item_stable>_')
                        rule['stable'].append({'attribute': val[0], 'value': val[1]})
                    else:
                        val = item.split('_<item_flexible>_')
                        rule['stable'].append({'attribute': val[0], 'value': val[1], 'flexible_as_stable': True})
                else:
                    val = item.split('_<item_flexible>_')
                    val_desired = ar_dict['desired']['itemset'][i].split('_<item_flexible>_')
                    rule['flexible'].append({'attribute': val[0], 'undesired': val[1], 'desired': val_desired[1]})
            rule['target'] = {
                'attribute': self.target,
                'undesired': ar_dict['undesired']['target'].split('_<item_target>_')[1],
                'desired': ar_dict['desired']['target'].split('_<item_target>_')[1],
            }
            rule['support of undesired part'] = int(ar_dict['undesired']['support'])
            rule['confidence of undesired part'] = float(ar_dict['undesired']['confidence'])
            rule['support of desired part'] = int(ar_dict['desired']['support'])
            rule['confidence of desired part'] = float(ar_dict['desired']['confidence'])
            rule['uplift'] = float(ar_dict['uplift'])
            rules.append(rule)
        return json.dumps(rules)

    def get_pretty_ar_notation(self):
        """
        Generate a list of text strings representing the action rules.

        Returns
        -------
        list
            List of text strings representing the action rules.
        """
        rules = []
        for ar_dict in self.action_rules:
            text = "If "
            for i, item in enumerate(ar_dict['undesired']['itemset']):
                if item == ar_dict['desired']['itemset'][i]:
                    if '_<item_stable>_' in item:
                        val = item.split('_<item_stable>_')
                        text += "attribute '" + val[0] + "' is '" + val[1] + "', "
                    else:
                        val = item.split('_<item_flexible>_')
                        text += "attribute (flexible is used as stable) '" + val[0] + "' is '" + val[1] + "', "
                else:
                    val = item.split('_<item_flexible>_')
                    val_desired = ar_dict['desired']['itemset'][i].split('_<item_flexible>_')
                    text += "attribute '" + val[0] + "' value '" + val[1] + "' is changed to '" + val_desired[1] + "', "
            text += (
                "then '"
                + self.target
                + "' value '"
                + ar_dict['undesired']['target']
                + "' is changed to '"
                + ar_dict['desired']['target']
                + " with uplift: "
                + str(ar_dict['uplift'])
                + "."
            )
            rules.append(text)
        return rules
