import aif360
import pandas as pd
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset, StandardDataset

class datasets_loader:


    dataset = None


    def default_preprocessing_german(df):
        """Adds a derived sex attribute based on personal_status."""
        # TODO: ignores the value of privileged_classes for 'sex'
        status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                    'A92': 'female', 'A95': 'female'}
        df['sex'] = df['personal_status'].replace(status_map)

        return df

    #Genero
    def load_german_dataset(self):

        default_mappings = {
        'label_maps': [{1.0: 'Good Credit', 0.0: 'Bad Credit'}],
        'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}],
        }

        self.dataset = GermanDataset(
        protected_attribute_names=['sex'],         
        privileged_classes=[['male']],      # age >=25 is considered privileged
        features_to_drop=['personal_status','sex'],
        categorical_features=['status', 'credit_history', 'purpose',
                     'savings', 'employment', 'other_debtors', 'property',
                     'installment_plans', 'housing', 'skill_level', 'telephone',
                     'foreign_worker'],
        metadata=default_mappings,custom_preprocessing=default_preprocessing_german
        )


    # idade
    def load_bank_dataset(self):
    
        self.dataset= BankDataset(
        label_name='y', favorable_classes=['yes'],
                     protected_attribute_names=['age'],
                     privileged_classes=[lambda x: x >= 25],
                     instance_weights_name=None,
                     categorical_features=['job', 'marital', 'education', 'default',
                         'housing', 'loan', 'contact', 'month', 'day_of_week',
                         'poutcome'],
                     features_to_keep=[], features_to_drop=['age'],
                     na_values=["unknown"], custom_preprocessing=None,
                     metadata=None)


    # Raça
    def load_adult_dataset(self):

    
        default_mappings = {
            'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
            'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'}]
            }
        self.dataset = AdultDataset(label_name='income-per-year',
                    favorable_classes=['>50K', '>50K.'],
                    protected_attribute_names=['race'],
                    privileged_classes=[['White']],
                    instance_weights_name=None,
                    categorical_features=['workclass', 'education',
                        'marital-status', 'occupation', 'relationship',
                        'native-country'],
                    features_to_keep=[], features_to_drop=['fnlwgt','race'],
                    na_values=['?'], custom_preprocessing=None,
                    metadata=default_mappings)


    def default_preprocessing_compas(df):
        """Perform the same preprocessing as the original analysis:
        https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        """
        return df[(df.days_b_screening_arrest <= 30)
                & (df.days_b_screening_arrest >= -30)
                & (df.is_recid != -1)
                & (df.c_charge_degree != 'O')
                & (df.score_text != 'N/A')]



    # Raça    
    def load_compas_dataset(self):
    

        default_mappings = {
            'label_maps': [{0.0: 'Did recid.', 1.0: 'No recid.'}],
            'protected_attribute_maps': [{1.0: 'Caucasian', 0.0: 'Not Caucasian'}]
        }
    
        self.dataset = CompasDataset(label_name='two_year_recid', favorable_classes=[0],
                 protected_attribute_names=[ 'race'],
                 privileged_classes=[['Female'], ['Caucasian']],
                 instance_weights_name=None,
                 categorical_features=['age_cat', 'c_charge_degree',
                     'c_charge_desc'],
                 features_to_keep=['sex', 'age', 'age_cat',
                     'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                     'priors_count', 'c_charge_degree', 'c_charge_desc',
                     'two_year_recid'],
                 features_to_drop=['race'], na_values=[],
                 custom_preprocessing=default_preprocessing_compas,
                 metadata=default_mappings)


    def load_titanic_dataset(self):

        df = pd.read_csv('bases/titanic/train.csv')
        default_mappings = {
        'label_maps': [{1.0: 1, 0.0: 0}],
        'protected_attribute_maps': [{0.0: 'male', 1.0: 'female'}]
        }
        
        self.dataset = StandardDataset(df,label_name='Survived',
                 favorable_classes=[1],
                 protected_attribute_names=['Sex'],
                 privileged_classes=[['female']],
                 instance_weights_name=None,
                 categorical_features=['Name','Ticket','Cabin','Embarked'],
                 features_to_keep=[], features_to_drop=['PassengerId','Sex'],
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings)



def main():
    d = datasets_loader()
    datasets = {'german':d.load_german_dataset(),
            'bank':d.load_bank_dataset(),
            'adult':d.load_adult_dataset(),
            'compas':d.load_compas_dataset(),
            'titanic':d.load_titanic_dataset()}

    for name, loader in datasets.items():
    
        dataset_orig = loader
        print(name)


if __name__ == "__main__":
    main()

    


    

    




    

    





