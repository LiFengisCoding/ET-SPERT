import argparse
import torch
import uer.trainer as trainer
from uer.utils.config import load_hyperparam
from uer.opts import *


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 预处理数据集的路径
    parser.add_argument("--dataset_path", type=str, default="dataset.pt",
                        help="Path of the preprocessed dataset.")
    # 词汇文件的路径
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    # 句段模型路径
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    # 目标词汇文件的路径
    parser.add_argument("--tgt_vocab_path", default=None, type=str,
                        help="Path of the target vocabulary file.")
    # 目标句段模型的路径
    parser.add_argument("--tgt_spm_model_path", default=None, type=str,
                        help="Path of the target sentence piece model.")
    # 预训练模型的路径
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path of the pretrained model.")
    # 输出模型的路径
    parser.add_argument("--output_model_path", type=str, required=True,
                        help="Path of the output model.")
    # 模型超参数配置文件
    parser.add_argument("--config_path", type=str, default="D:/PyCharm/Code/ET-SPERT/models/bert/base_config.json",
                        help="Config file of model hyper-parameters.")

    # Training and saving options
    # 总训练步骤
    parser.add_argument("--total_steps", type=int, default=100000,
                        help="Total training steps.")
    # 保存模型检查点的具体步骤
    parser.add_argument("--save_checkpoint_steps", type=int, default=10000,
                        help="Specific steps to save model checkpoint.")
    # 打印提示的具体步骤
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    # 累积梯度的具体步骤
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Specific steps to accumulate gradient.")
    # 训练批量大小。 实际的 batch_size 是 [batch_size乘以world_size乘以accumulation_steps]
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size. The actual batch_size is [batch_size x world_size x accumulation_steps].")
    # 实例在内存中的缓冲区大小
    parser.add_argument("--instances_buffer_size", type=int, default=25600,
                        help="The buffer size of instances in memory.")
    # 预测标签的数量
    parser.add_argument("--labels_num", type=int, required=False,
                        help="Number of prediction labels.")
    # dropout值
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value.")
    # 随机种子
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")

    # Preprocess options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Model options.
    model_opts(parser)
    # 目标嵌入类型
    parser.add_argument("--tgt_embedding", choices=["word", "word_pos", "word_pos_seg", "word_sinusoidalpos"],
                        default="word_pos_seg",
                        help="Target embedding type.")
    # 解码器类型
    parser.add_argument("--decoder", choices=["transformer"], default="transformer", help="Decoder type.")
    # 池化类型
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    # 预训练模型的训练目标
    parser.add_argument("--target", choices=["bert", "lm", "mlm", "bilm", "albert", "seq2seq", "t5", "cls", "prefixlm"],
                        default="bert",
                        help="The training target of the pretraining model.")
    # 绑定词嵌入和 softmax 权重
    parser.add_argument("--tie_weights", action="store_true",
                        help="Tie the word embedding and softmax weights.")
    # 在 lm 目标的 output_layer 上添加偏差
    parser.add_argument("--has_lmtarget_bias", action="store_true",
                        help="Add bias on output_layer for lm target.")

    # Masking options.
    # 全词掩蔽
    parser.add_argument("--whole_word_masking", action="store_true", help="Whole word masking.")
    # 跨度掩蔽
    parser.add_argument("--span_masking", action="store_true", help="Span masking.")
    # 跨度掩蔽的几何分布超参数
    parser.add_argument("--span_geo_prob", type=float, default=0.2,
                        help="Hyperparameter of geometric distribution for span masking.")
    # 跨度掩码的最大长度
    parser.add_argument("--span_max_length", type=int, default=10,
                        help="Max length for span masking.")

    # Optimizer options.
    optimization_opts(parser)

    # GPU options.
    # 用于训练的进程总数 (GPU)
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes (GPUs) for training.")
    # help="每个进程的等级列表。"
    #      “每个进程都有一个唯一的整数等级，其值在区间 [0, world_size) 内，并在单个 GPU 中运行。”）
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int, help="List of ranks of each process."
                                                                             " Each process has a unique integer rank whose value is in the interval [0, world_size), and runs in a single GPU.")
    # 用于训练的 master 的 IP-端口
    parser.add_argument("--master_ip", default="tcp://localhost:12345", type=str,
                        help="IP-Port of master for training.")
    # 分布式后端
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="gloo", type=str, help="Distributed backend.")

    args = parser.parse_args()

    if args.target == "cls":
        assert args.labels_num is not None, "Cls target needs the denotation of the number of labels."

    # 从配置文件加载超参数
    if args.config_path:
        load_hyperparam(args)

    ranks_num = len(args.gpu_ranks)
    if args.world_size > 1:
        assert torch.cuda.is_available(), "No available GPUs."
        assert ranks_num <= args.world_size, "Started processes exceed `world_size` upper limit."
        assert ranks_num <= torch.cuda.device_count(), "Started processes exceeds the available GPUs."
        args.dist_train = True
        args.ranks_num = ranks_num
        print("Using distributed mode for training.")
    elif args.world_size == 1 and ranks_num == 1:
        # Single GPU mode.
        assert torch.cuda.is_available(), "No available GPUs."
        args.gpu_id = args.gpu_ranks[0]
        assert args.gpu_id < torch.cuda.device_count(), "Invalid specified GPU device."
        args.dist_train = False
        args.single_gpu = True
        print("Using GPU %d for training." % args.gpu_id)
    else:
        # CPU mode.
        assert ranks_num == 0, "GPUs are specified, please check the arguments."
        args.dist_train = False
        args.single_gpu = False
        print("Using CPU mode for training.")

    trainer.train_and_validate(args)


if __name__ == "__main__":
    main()
