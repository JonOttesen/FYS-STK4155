import numpy as np

class latex_print():

    def __init__(self, X, column_text = None, row_text = None, errors = None, decimal = 3, num = 1, caption = ''):

        if len(X.shape) > 1:
            self.X = X
        else:
            self.X = np.reshape(X, (len(X), 1))

        self.row_text = row_text
        self.column_text = column_text
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
        self.num = num
        self.caption = caption

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

        for k in range(len(self.rounded)):
            for i in range(len(self.rounded[0])):
                if i == 0:
                    try:
                        printing += str(self.row_text[k]) + ' & '
                    except:
                        pass
                if self.errors_rounded[k,i] == 0:
                    printing += str(self.rounded[k, i]) + ' & '
                else:
                    printing += str(self.rounded[k, i]) + r' \(\pm\) ' + str(self.errors_rounded[k,i]) + ' & '

            printing = printing[:-2]
            printing += '\\\ \hline\n'

        return printing

    def table_start(self):
        printing = ''
        printing += r'\begin{table}[H]' + '\n' + r'\begin{tabular}'
        printing += r'{'

        for i in range(len(self.X[0]) + (0 if type(self.row_text) == type(None) else 1)):
            printing += r'|c'
        printing += r'|}\hline' + '\n'

        for i in range(len(self.column_text) if type(self.column_text) != type(None) else 0):
            printing += str(self.column_text[i]) + ' & '

        if type(self.column_text) != type(None):
            printing = printing[:-2]
            printing += '\\\ \hline\n'

        return printing

    def table_end(self):
        printing = ''
        printing += r'\end{tabular}' + '\n'
        printing += r'\caption{' + str(self.caption) + '}' + '\n'
        printing += r'\label{tab:' + '{:02d}'.format(self.num) + '}' + '\n'
        printing += r'\end{table}' + '\n'
        return printing



    def __str__(self):
        return self.table_start() + self.table_text + self.table_end()

if __name__ == "__main__":
    test = latex_print(np.random.normal(0, 100, (5, 35)), decimal = 4)
    print(test)


"""
\begin{table}
\begin{tabular}{|c|c|c|c|}\hline
\(\lambda\)-values & Fold 1 & Fold 2 & Fold 3 \\ \hline
\(\lambda_{1}\) & 1 & 2 & 2\\ \hline
\(\lambda_{2}\) & 2 & 2 & 2 \\ \hline
\(\lambda_{3}\) & 1 & 2 & 1 \\ \hline
\(\lambda_{4}\) & 1 & 1.5 & 1 \\ \hline
\end{tabular}
\caption{A illustration of how the error estimate can differ depending on the fold and the \(\lambda\) value. The ideal \(\lambda\) would here be \(\lambda_4\).}
\label{tab:01}
\end{table}
"""































#jao
