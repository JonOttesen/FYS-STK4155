import numpy as np

class latex_print():

    def __init__(self, X, text = None, errors = None, decimal = 3):

        if len(X.shape) > 1:
            self.X = X
        else:
            self.X = np.reshape(X, (1, len(X)))

        self.text = text
        if isinstance(errors, (list, np.ndarray)):
            if isinstance(errors, list):
                errors = np.array(errors)

            if np.shape(self.X) != np.shape(errors):
                self.errors = np.reshape(errors, self.X.shape)
            else:
                self.errors = errors
        else:
            self.errors = np.zeros_like(self.X)

        self.decimal = decimal
        self.rounded = self.rounding(self.X)
        self.errors_rounded = self.rounding(self.errors)

        self.table_text = self.table_print()

    def rounding(self, X):
        Rounded = np.zeros_like(X)
        Z = np.copy(X)
        Z[X == 0] = 1
        steps = np.linspace(-8, int(np.log10(np.max(Z))), int(np.log10(np.max(Z))) + 9)

        for i in steps:
            indexes = np.logical_and(i < np.log10(np.abs(Z)), np.log10(np.abs(Z)) < i + 1)
            Rounded[indexes] = np.round(X[indexes], - int(i) + self.decimal - 1)

        return Rounded


    def table_print(self):
        printing = ''

        for i in range(len(self.rounded[0])):
            for k in range(len(self.rounded)):
                if k == 0:
                    try:
                        printing += self.text[i] + ' & '
                    except:
                        pass
                if self.errors_rounded[k,i] == 0:
                    printing += str(self.rounded[k, i]) + ' & '
                else:
                    printing += str(self.rounded[k, i]) + r' \(\pm\) ' + str(self.errors_rounded[k,i]) + ' & '

            printing = printing[:-2]
            printing += '\\\ \hline\n'

        return printing

    def __str__(self):
        return self.table_text

if __name__ == "__main__":
    test = latex_print(np.random.normal(0, 100, (5, 35)), decimal = 4)
    print(test)
































#jao
