from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, 
    QPushButton, QHBoxLayout, QMessageBox
)
from .fft import apply_fft

class FFTDialog(QDialog):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fourier Transform Analysis")
        self.resize(350, 200)
        self.state = state
        self.parent_viewer = parent
        
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                color: #000000;
            }
            QPushButton {
                background-color: #f0f0f0;
                color: #000000;
                border: 1px solid #999999;
                border-radius: 3px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #e5f1fb;
                border: 1px solid #0078d7;
            }
            QComboBox {
                border: 1px solid #999999;
                background-color: white;
                color: black;
                padding: 3px;
            }
        """)

        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["amplitude", "phase", "real", "imaginary"])
        
        self.window_combo = QComboBox()
        self.window_combo.addItems(["none", "sine", "kaiser", "gauss", "blackmanharris"])
        
        self.dir_combo = QComboBox()
        self.dir_combo.addItems(["Forward (FT)", "Inverse (iFT)"])
        
        form_layout.addRow("Output Component:", self.type_combo)
        form_layout.addRow("Window Function:", self.window_combo)
        form_layout.addRow("Transform Direction:", self.dir_combo)
        
        layout.addLayout(form_layout)
        
        btn_layout = QHBoxLayout()
        self.calc_btn = QPushButton("Compute & Show")
        self.calc_btn.clicked.connect(self.compute_fft)
        btn_layout.addStretch()
        btn_layout.addWidget(self.calc_btn)
        
        layout.addLayout(btn_layout)

    def compute_fft(self):
        try:
            w_type = self.window_combo.currentText()
            o_type = self.type_combo.currentText()
            direction = 'ft' if 'Forward' in self.dir_combo.currentText() else 'ift'
            
            result = apply_fft(self.state.data, window_type=w_type, output_type=o_type, direction=direction)
            
            # Avoid circular import by importing locally
            from .api import img_viewer
            
            # Open a new viewer for the result
            viewer = img_viewer(result, colormap_name=self.state.colormap_name)
            
            if self.parent_viewer:
                if not hasattr(self.parent_viewer, 'child_viewers'):
                    self.parent_viewer.child_viewers = []
                self.parent_viewer.child_viewers.append(viewer)
                
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "FFT Error", f"Failed to compute FFT:\n{str(e)}")
