from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

@app.route('/training', methods=['POST'])
def training():
    try:
      data = request.get_json()  # Obtém o array do corpo da requisição
      new_test = data['data']  # Assumindo que o array está em um campo chamado 'data'

      path_training = 'treinamento.pkl'

      if os.path.exists(path_training):
        cart_from_joblib = joblib.load(path_training)
        predict = cart_from_joblib.predict([new_test])

        return jsonify({"id_profile": predict[0].tolist()})  # Convertendo para lista Python e retornando como JSON

      else:
        return jsonify({"error": "Arquivo 'treinamento.pkl' não encontrado."}), 404

    except Exception as e:
      return jsonify({"error": f"Erro ao realizar o treinamento: {e}"}), 500

if __name__ == '__main__':
  app.run(debug=True, port=3008)
