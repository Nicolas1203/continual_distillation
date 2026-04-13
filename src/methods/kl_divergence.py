from src.methods.base import BaseMethod
from src.utils.train_functions import train_kl_divergence


class KLDivergenceMethod(BaseMethod):
    method_name = "kl_divergence"

    def train(self, *, student, domains_teacher, domains_data, device, epochs, dataloader, num_classes):
        return train_kl_divergence(
            student=student,
            optimizer=self.optimizer,
            domains_teacher=domains_teacher,
            domains_data=domains_data,
            device=device,
            epochs=epochs,
            args=self.args,
            dataloader=dataloader,
            num_classes=num_classes,
        )
