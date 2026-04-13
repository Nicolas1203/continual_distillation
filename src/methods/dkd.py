from src.methods.base import BaseMethod
from src.utils.train_functions import train_dkd


class DKDMethod(BaseMethod):
    method_name = "dkd"

    def train(self, *, student, domains_teacher, domains_data, device, epochs, dataloader, num_classes):
        return train_dkd(
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
