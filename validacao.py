# Importando pacotes e funções
import os
import time
import datetime
import random
import torch
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

nrows = 2000
lotes = [16,32]
taxas_aprendizado = [2e-5, 3e-5, 5e-5]
epocas = [2,3,4]

def executa_validacao(lote, taxa, epoca, nrow):
    # Declarando constantes
    BANCO_FINAL = r"dados/data.csv"
    PASTA_MODELO = f"dados/model_save_l{lote}_t{taxa}_e{epoca}/"

    device = torch.device("cpu")
    os.environ['CURL_CA_BUNDLE'] = ''  # Caso surja o erro "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed" rode este comando e reduza a versão do pacote requets: "pip uninstall requests" e "pip install requests==2.27.1"


    # Declarando a semente aleatória para que este código seja reprodutível
    my_seed = 288933

    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)

    df = pd.read_csv(BANCO_FINAL, nrows=nrow)

    model_name = 'bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels = 2, output_attentions = False, output_hidden_states = False)

    # Tokeniza todas as sentenças utilizando o método wordpiece
    input_ids = []
    attention_masks = []

    for sentenca in df['titulo'].values:
        encoded_dict = tokenizer.encode_plus(
                            sentenca,
                            add_special_tokens = True,  # Inclui os tokens '[CLS]' e '[SEP]' no início e fim da lista, respectivamente
                            max_length = 64,  # Sentenças maiores do que 64 tokens serão truncadas e sentenças menores terão o restante da lista preenchida com o token '[PAD]'
                            truncation=True,  # Para usar o truncamento
                            padding='max_length',  # Para incluir o token '[PAD]' se necessario
                            return_attention_mask = True,  # Retorna a lista de atenção, que é 1 em na posição em que o token não for '[PAD]' e é 0 quando o token for '[PAD]'
                            return_tensors = 'pt'  # Retorna as listas como objeto do tipo tensor
                    )

        # Salva a sentença tokenizada e a lista de atenção em uma lista
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Transforma a lista de tensores em um tensor com múltiplas listas
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['label'].values)

    # Exemplificando o resultado com a primeira sentença do banco
    print('Sentença original: ', df['titulo'].values[0])
    print('Sentença tokenizada: ', input_ids[0])
    print('Atenção: ', attention_masks[0])
    print('Label: ', labels[0])

    # Combinando as listas obtidas na fase de tokenização para criar um objeto TensorDataset
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Calculando o número de sentenças que cada conjunto precisa ter para seguir a proporção 80-10-10
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Divindo os conjuntos aleatoriamente de acordo com o tamanho amostral calculado
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    print('{:>5,} sentenças para treinamento'.format(train_size))
    print('{:>5,} sentenças para validação'.format(val_size))
    print('{:>5,} sentenças para teste'.format(test_size))

    batch_size = lote
    learning_rate = taxa
    epochs = epoca

    # Separando os conjuntos considerando o tamanho do lote:
    # Para o treinamento, selecionamos os batches aleatoriamente. Já para validação e teste, selecionamos os batches sequencialmente
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
    validation_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = batch_size)
    prediction_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = batch_size)

    # O otimizador é o objeto responsável por realizar a atualização dos parâmetros do modelo
    # Vamos utilizar o otimizador AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    # Cria o objeto responsável por diminuir a taxa de aprendizagem linearmente na medida em que o modelo aprende
    # O 'num_training_steps' é calculado como o número de lotes multiplicado pelo número de épocas
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * epochs)

    # Função que calcula a acurácia dos valores preditos contra os labels
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(elapsed):
        elapsed_rounded = int(round((elapsed)))

        return str(datetime.timedelta(seconds=elapsed_rounded))

    training_stats = []  # Vamos armazenar algumas estatísticas de avaliação do modelo neste objeto

    tempo_inicial = time.time()

    # Para cada época...
    for epoch_i in range(0, epochs):
        # ========================================
        #               Treinamento
        # ========================================

        # Realiza uma passagem completa sobre o conjunto de treinamento

        print('\n======== Epoca {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Treinando...')
        
        t0 = time.time()  # Mede o tempo de treinamento desta época
        total_train_accuracy = 0  # Reinicia a acuracia total para esta época
        total_train_loss = 0  # Reinicia a perda total para esta época

        model.train()  # Coloca o modelo em modo de treinamento
        # O método 'train()' apenas muda o estado do objeto 'modelo', ele não treina o modelo
        # As camadas 'dropout' e 'batchnorm' se comportam de maneira diferente durante o treinamento
        # em relação ao teste (https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        
        # Para cada lote (batch) dos dados de treinamento...
        for step, batch in enumerate(train_dataloader):
            # Imprimi o progresso para o usuário a cada 40 lotes
            if step % 40 == 0 and not step == 0:
                print('  Lote {:>5,}  de  {:>5,}.    Tempo decorrido: {:}.'.format(step, len(train_dataloader), format_time(time.time() - t0)))

            # 'batch' é um objeto que contém 3 tensores:
            #   [0]: input_ids
            #   [1]: attention_masks
            #   [2]: labels
            # Vamos copiar cada tensor para a CPU utilizando o método 'to()'
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()  # Precisamos limpar quaisquer gradientes calculados anteriormente antes de realizar o passo backward
            # O PyTorch não faz isso automaticamente devido a características do treinamento de RNN's
            # (https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            
            result = model(b_input_ids,
                        token_type_ids=None,  # 'None' porque não estamos no contexto de next sentence prediction
                        attention_mask=b_input_mask,
                        labels=b_labels,
                        return_dict=True)
            # Realiza o treinamento do modelo com os dados deste lote, utilizando a função forward
            # Embora utilizamos a função 'model', os argumentos desta função estão indo para outra função chamada 'forward'
            # A documentação dos resultados retornados está aqui:
            # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput

            loss = result.loss  # Perda de classificação. O quão "longe" a predição do modelo foi em relação ao label verdadeiro
            logits = result.logits  # Escore de classificação
            # Para cada sentença, o label com o maior escore será aquele que o modelo escolherá como o label verdadeiro da sentença

            total_train_loss += loss.item()  # Acumulando o erro de treinamento em todos os lotes para que possamos calcular a perda média ao final
            
            # Movendo os logits e labels para a CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_train_accuracy += flat_accuracy(logits, label_ids)  # Acumula a acuracia de treinamento

            loss.backward()  # Executa o método backward para calcular os gradientes

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Isto evita o problema de gradiente explosivo

            optimizer.step()  # Atualiza os parâmetros do modelo e executa um passo utilizando o gradiente computado
            
            scheduler.step()  # Atualiza a taxa de aprendizado
        
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)  # Calcula a acuracia media de treinamento
        print("  Acuracia: {0:.2f}".format(avg_train_accuracy))

        avg_train_loss = total_train_loss / len(train_dataloader)  # Calcula a perda média dos lotes

        training_time = format_time(time.time() - t0)

        print("  Perda media de treinamento: {0:.2f}".format(avg_train_loss))
        print("  Treinamento da epoca levou: {:}".format(training_time))

        # ========================================
        #               Validação
        # ========================================
        # Finalizada a época de treinamento, vamos medir a performance do modelo nos dados de validação

        print("\nValidando...")

        t0 = time.time()

        model.eval()  # Coloca o modelo em modo de validação

        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in validation_dataloader:
            # Similar ao treinamento, vamos armazenar os valores do lote
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # 'torch.no_grad()' economiza tempo de processamento por não computar algumas métricas que só são usadas na fase de treinamento
            with torch.no_grad():
                result = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels,
                            return_dict=True)
                # Executa a função 'forward' com os dados de validação

            loss = result.loss
            logits = result.logits

            total_eval_loss += loss.item()  # Acumulando o erro de validação em todos os lotes para que possamos calcular a perda média ao final

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)  # Acumula a acuracia de validação


        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)  # Calcula a acuracia media de validação
        print("  Acuracia: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)  # Calcula a perda média dos lotes

        validation_time = format_time(time.time() - t0)

        print("  Perda de validacao: {0:.2f}".format(avg_val_loss))
        print("  Validacao levou: {:}".format(validation_time))

        # Guardando estatísticas desta época
        training_stats.append(
            {
                'epoca': epoch_i + 1,
                'Perda de treinamento': avg_train_loss,
                'Perda de validacao': avg_val_loss,
                'Acuracia de treinamento': avg_train_accuracy,
                'Acuracia de validacao': avg_val_accuracy,
                'Tempo de treinamento': training_time,
                'Tempo de validacao': validation_time
            }
        )

    print("\nTreinamento concluído!")
    print("Tempo total de treinamento: {:} (hh:mm:ss)".format(format_time(time.time()-tempo_inicial)))

    # Salvando o modelo treinado e o tokenizador
    if not os.path.exists(PASTA_MODELO):
        os.makedirs(PASTA_MODELO)

    # Salvando as estatísticas do modelo
    df_stats = pd.DataFrame(data=training_stats).set_index('epoca')
    df_stats.to_csv(os.path.join(PASTA_MODELO, 'training_stats.csv'), index=False)

    validacao_dict = training_stats[-1]
    validacao_dict['lote'] = lote
    validacao_dict['taxa'] = taxa
    validacao_dict['epoca'] = epoca
    validacao_dict['observacoes'] = nrow

    return validacao_dict

validacao_final = []
for lote in lotes:
    for taxa in taxas_aprendizado:
        for epoca in epocas:
            validacao_final.append(executa_validacao(lote=lote, taxa=taxa, epoca=epoca, nrow=nrows))

df_stats = pd.DataFrame(data=validacao_final)
df_stats.to_csv("validacao.csv", index=False, sep=';')
