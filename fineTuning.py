from transformers import TrainingArguments, CLIPModel
from clipTrainer import ClipTrainer
from utils.pathUtils import prepare_output_path, get_checkpoint_path
from prepare_dataset import Builder
from utils.commonUtils import start_training
from transformers.training_args import OptimizerNames
import warnings

warnings.filterwarnings('ignore')

IGNORE_KEYS = []


def get_fine_tuning_trainer_args(output_path, hyperparameters):

    return TrainingArguments(
        output_dir=output_path + 'training/',
        logging_dir=output_path + 'logs/',
        per_device_train_batch_size=hyperparameters.TrainBatchSize,
        per_device_eval_batch_size=hyperparameters.EvalBatchSize,
        eval_strategy="no", # steps
        num_train_epochs=hyperparameters.Epochs,
        save_steps=hyperparameters.Steps.SaveSteps,
        eval_steps=hyperparameters.Steps.EvalSteps,
        logging_steps=hyperparameters.Steps.LoggingSteps,
        learning_rate=hyperparameters.Lr,
        lr_scheduler_type='cosine',
        warmup_ratio=hyperparameters.WarmUpRatio,
        weight_decay=hyperparameters.WeightDecay,
        save_total_limit=2,
        metric_for_best_model='bleu',
        greater_is_better=True,
        optim=OptimizerNames.ADAMW_HF,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=False, # True
        seed=42,
        half_precision_backend="auto",
        gradient_accumulation_steps=hyperparameters.Steps.GradientAccumulation,
        use_cpu=True,
    )


def train(Args):

    builder = Builder(Args)

    output_path = prepare_output_path('FineTuned', Args)

    fine_tune_args = get_fine_tuning_trainer_args(output_path, Args.FineTuning.Hyperparameters)

    if Args.FineTuning.Model.LoadCheckPoint:
        model_path = get_checkpoint_path('FineTuned', Args)
    else:
        model_path = Args.FineTuning.Model.Name

    model = CLIPModel.from_pretrained(model_path, cache_dir=Args.FineTuning.Model.CachePath)

    training_data = builder.get_train_dataset()
    # testing_data = builder.get_eval_dataset()

    fine_tune_trainer = ClipTrainer(
        model=model,
        configArgs=Args,
        args=fine_tune_args,
        # compute_metrics=compute_metrics,
        data_collator=builder.get_collate_fn(),
        train_dataset=training_data,
        # eval_dataset=testing_data
    )

    start_training(Args, fine_tune_trainer, Args.FineTuning.Model.LoadCheckPoint, model_path, output_path,
                   Args.FineTuning.Model.OutputPath, training_data)