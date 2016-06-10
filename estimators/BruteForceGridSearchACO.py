from BruceForceGridSearchBase import *
class BruteForceGridSearch(BruteForceGridSearchBase):
    def fit(self,data=None):
        n_hidden= np.array([55])
        result = []
        for train_item in self.train_bucket:
            X_train, y_train, X_test, y_test = train_item.getitems()
            n_hidden = np.array([130])
            param_aco = {
                'Q': [0.65],
                'epsilon': [0.1],
                'hidden_nodes': [n_hidden]
            }
            estimator = ACOEstimator(Q=0.65, epsilon=0.1, number_of_solutions=130)
            neuralNet = NeuralFlowRegressor(learning_rate=1E-03, hidden_nodes=n_hidden)
            neural_shape = [X_train.shape[1], n_hidden[0], y_train.shape[1]]
            fit_param = {'neural_shape': neural_shape}
            gridSearch = GridSearchCV(estimator,param_aco,fit_params=fit_param,n_jobs=-1)
            gridSearch.fit(X_train,y_train)
            optimizer = OptimizerNNEstimator(gridSearch.best_estimator_, neuralNet)
            optimizer.fit(X_train,y_train,**fit_param)
            X_test_f = self.data_source[self.train_len+1:self.train_len+self.test_len+1]
            y_pred_f = optimizer.predict(X_test)
            y_pred = self.fuzzy_transform.defuzzy(X_test_f,y_pred_f)
            score_nn = np.sqrt(mean_squared_error(y_test[1:], y_pred))
            tmp = {
                'score':score_nn,
                'n_sliding':train_item.metadata['sliding_windows'],
                'best_estimator':'%s'%gridSearch.best_estimator_
            }
            result.append(tmp)
            np.savez('../model_saved/%s'%score_nn,y_pred=y_pred,y_test=y_test[1:])
            optimizer.save('../model_saved/%s_model'%score_nn)
            pd.DataFrame(result).to_csv('score_grid_exhaust.csv',index=None)

