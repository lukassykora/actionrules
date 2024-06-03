"""Class Output."""


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
        rule = '['
        for i, item in enumerate(self.action_rules['undesired']['itemset']):
            if i > 0:
                rule += ' ∧ '
            rule += '('
            if item == self.action_rules['desired']['itemset'][i]:
                val = item.split('_<item>_')
                rule += str(val[0]) + ': ' + str(val[1])
            else:
                val = item.split('_<item>_')
                val_desired = self.action_rules['desired']['itemset'][i].split('_<item>_')
                rule += str(val[0]) + ': ' + str(val[1]) + ' → ' + str(val_desired[1])
            rule += ')'
        rule += (
            '] ⇒ ['
            + str(self.target)
            + ': '
            + str(self.action_rules['undesired']['target'])
            + ' → '
            + str(self.action_rules['desired']['target'])
            + ']'
        )
        rule += (
            ', support of undesired part: '
            + str(self.action_rules['undesired']['support'])
            + ', confidence of undesired part: '
            + str(self.action_rules['undesired']['confidence'])
        )
        rule += (
            ', support of desired part: '
            + str(self.action_rules['desired']['support'])
            + ', confidence of desired part: '
            + str(self.action_rules['desired']['confidence'])
        )
        return rule

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
                    val = item.split('_<item>_')
                    rule['stable'].append({'attribute': val[0], 'value': val[1]})
                else:
                    val = item.split('_<item>_')
                    val_desired = ar_dict['desired']['itemset'][i].split('_<item>_')
                    rule['flexible'].append({'attribute': val[0], 'undesired': val[1], 'desired': val_desired[1]})
            rule['target'] = {
                'attribute': self.target,
                'undesired': ar_dict['undesired']['target'],
                'desired': ar_dict['desired']['target'],
            }
            rule['support of undesired part'] = ar_dict['undesired']['support']
            rule['confidence of undesired part'] = ar_dict['undesired']['confidence']
            rule['support of desired part'] = ar_dict['desired']['support']
            rule['confidence of desired part'] = ar_dict['desired']['confidence']
            rules.append(rule)
        return rules
