from src.methods.base import BaseMethod
from src.utils.train_functions import train_self_distillation


class SelfDistillationMethod(BaseMethod):
    method_name = "self_distillation"

    def train(self, *, student, domains_teacher, domains_data, device, epochs, dataloader, num_classes, old_student=None):
        return train_self_distillation(
            student=student,
            optimizer=self.optimizer,
            domains_teacher=domains_teacher,
            domains_data=domains_data,
            device=device,
            epochs=epochs,
            args=self.args,
            old_student=old_student,
            dataloader=dataloader,
            num_classes=num_classes,
        )
