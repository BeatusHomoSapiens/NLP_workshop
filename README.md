NLP интенсив: задача множественной классификации обратной связи от пользователей

16 место лидерборда, ник Seidel, результат: 0.5024379433

Структура проекта:
* ./data - папка с данными для обучения и тестирования  
  * train.csv - исходный .csv файл для обучения
  * test.csv - исходный .csv файл для тестирования
* bert.ipynb - ноутбук для воспроизведения лучшего решения на лидерборде. Для воспроизведения запустить все ячейки в разделе ноутбука "BERT - best model recreation"
* datasets.py - файл с torch датасетами
* models.py - файл с torch моделями
* utils.py - файл с вспомогательными функциями
