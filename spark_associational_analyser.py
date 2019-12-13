import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

import findspark
findspark.init()
import pyspark
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SQLContext
from pyspark.sql.functions import desc

class SparkAssociationalAnalyser:
    
    def __init__(self):
        self.sc = pyspark.SparkContext(appName="AssociationalAnalysis")
        try:
            self.sqlContext = SQLContext(self.sc)
        except Exception as e: 
            print(e)
            self.sc.stop()
            
    def drop_singly_occuring_elements(self, df):
        print('Drop singly occuring elements...')
        singles = df['ITEMS'].apply(lambda x: x if len(x) == 1 else np.nan).dropna()
        drops = []
        with tqdm(total=len(df) * len(singles)) as pbar:
            for _, row in df.iterrows():
                l = row['ITEMS']
                for ind, s in singles.iteritems():
                    if (s != l) & (str(s[0]) in l):
                        drops.append(ind)
                    pbar.update(1)
        drops = list(set(drops))
        df = df.drop(drops)
        print('Dropped ' + str(len(drops)) + ' only singly occuring elements.')
        return df
    
    def prepare_data(self, df):
        df['ITEMS'] = df['ITEMS'].apply(lambda x: x.split(','))
        df = self.drop_singly_occuring_elements(df)
        expl = list()
        print('Explode DataFrame...')
        print('Number of Rows before Explosion: ' + str(len(df)))
        with tqdm(total=len(df)) as pbar:
            for _, row in df.iterrows():
                expl.append([row['ITEMS']]*row['FREQUENCY'])
                pbar.update(1)
        flatten = [item for sublist in expl for item in sublist]
        exploded = pd.DataFrame(columns={'items'})
        exploded['items'] = flatten
        print('Number of Rows after Explosion: ' + str(len(exploded)))
        print('Creating Spark DataFrame...')
        spark_df = self.sqlContext.createDataFrame(exploded)
        return spark_df

    def mine_association_rules(self, spark_df, min_support = 0.0001, min_confidence = 0.7):
        try:
            fpGrowth = FPGrowth( minSupport=min_support, minConfidence=min_confidence)
            print('Fitting FPGrowth model...')
            model = fpGrowth.fit(spark_df)
            associationRules = model.associationRules.orderBy(['confidence', 'lift'], ascending=False)
            associationRules._sc = self.sc
        except Exception as e: 
            print(e)
            self.sc.stop()
        print('Converting DataFrame...')
        return associationRules.toPandas()

    
    
    