import pandas as pd

data = pd.DataFrame({
    'Mes': range(1, 13),
    'Publicidad': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650],
    'Ventas': [200, 240, 280, 310, 370, 400, 430, 480, 510, 540, 600, 640]
})

data.to_csv('datos.csv', index=False)