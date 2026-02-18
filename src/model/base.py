class BaseModel:
    """Lớp cơ sở cho tất cả các mô hình dự báo"""
    def __init__(self):
        pass

    def fit(self, data):
        """Huấn luyện mô hình trên dữ liệu lịch sử"""
        raise NotImplementedError

    def predict(self, steps):
        """Dự báo 'steps' bước tiếp theo"""
        raise NotImplementedError