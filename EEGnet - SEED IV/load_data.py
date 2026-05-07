import numpy as np
import scipy.io as sio
import os

class SEEDIVLoader:
    def __init__(self):
        # 根据 ReadMe.txt 定义的 SEED-IV 标签 [cite: 1, 2]
        # 0: neutral, 1: sad, 2: fear, 3: happy
        self.SESSION_LABELS = {
            1: np.array([1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]),
            2: np.array([2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]),
            3: np.array([1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0])
        }
        self.EMOTION_MAP = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}

    def load_single_session(self, file_path, session_id):
        """
        加载单个 .mat 文件（一个 Session）
        :param file_path: .mat 文件的绝对路径
        :param session_id: 该文件对应的 session 编号 (1, 2, 或 3)
        :return: (features, labels) 
                 features: list (长度 24)，每个元素为 (62, time, 5) 的特征矩阵
                 labels: (24,) 标签
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")
        
        if session_id not in self.SESSION_LABELS:
            raise ValueError("session_id 必须是 1, 2 或 3")

        print(f"📦 正在加载 Session {session_id}: {os.path.basename(file_path)}")
        mat_data = sio.loadmat(file_path)
        
        session_features = []
        session_labels = []
        # SEED-IV 每个 session 包含 24 个 trial [cite: 1, 2]
        for i in range(1, 25):
            # 这里的 key 通常是 'de_LDS1', 'de_LDS2' ... 到 'de_LDS24'
            key = f"de_LDS{i}"
            if key in mat_data:
                # trial shape: (62, time_windows, 5 bands)
                session_features.append(mat_data[key])
                session_labels.append(int(self.SESSION_LABELS[session_id][i - 1]))
            else:
                print(f"⚠️ 警告: 键值 {key} 在文件中不存在，请检查数据格式。")

        return session_features, np.array(session_labels, dtype=int)

    def load_subject_all_sessions(self, file_list):
        """
        一次性加载某个被试的全部三个 session
        :param file_list: 包含三个 .mat 路径的列表，顺序需对应 session 1, 2, 3
        :return: 合并后的 features (list) 和 labels (ndarray)
        """
        all_features = []
        all_labels = []

        for idx, path in enumerate(file_list):
            session_id = idx + 1
            feat, lab = self.load_single_session(path, session_id)
            all_features.extend(feat)
            all_labels.append(lab)

        return all_features, np.concatenate(all_labels)


    def _find_subject_file_in_session(self, session_folder, subject_id):
        """
        在指定 session 文件夹中，自动匹配某个被试的 .mat 文件。
        规则：文件名以 "{subject_id}_" 开头且以 ".mat" 结尾，例如 "1_20160518.mat"
        若匹配到多个文件，按文件名排序后取第一个。
        """
        if not os.path.isdir(session_folder):
            raise FileNotFoundError(f"Session 文件夹不存在: {session_folder}")

        candidates = sorted([
            os.path.join(session_folder, fname)
            for fname in os.listdir(session_folder)
            if fname.startswith(f"{subject_id}_") and fname.endswith(".mat")
        ])

        if len(candidates) == 0:
            raise FileNotFoundError(
                f"未找到被试 {subject_id} 的数据文件：{session_folder} 下不存在以 '{subject_id}_' 开头的 .mat 文件"
            )

        if len(candidates) > 1:
            print(f"⚠️ 警告: 被试 {subject_id} 在 {session_folder} 匹配到多个文件，将使用: {os.path.basename(candidates[0])}")

        return candidates[0]

    def load_single_subject(self, subject_id, base_path):
        """
        加载某一个被试的 3 个 session（自动在 base_path/1,2,3 中查找对应 .mat）

        目录结构假设：
            base_path/
                1/   # session 1
                2/   # session 2
                3/   # session 3

        :param subject_id: 被试编号 (例如 1, 2, 3...)
        :param base_path: eeg_feature_smooth 根目录
        :return: (features, labels)
                 features: list（长度 72=24*3），每个元素为 (62, time, 5)
                 labels: ndarray (72,)
        """
        file_list = []
        for session_id in [1, 2, 3]:
            session_folder = os.path.join(base_path, str(session_id))
            file_path = self._find_subject_file_in_session(session_folder, subject_id)
            file_list.append(file_path)

        return self.load_subject_all_sessions(file_list)

    def load_all_subjects(self, base_path, subject_ids=None):
        """
        加载所有被试（或指定被试列表）的 3 个 session 数据。

        :param base_path: eeg_feature_smooth 根目录
        :param subject_ids:
            - None：自动扫描 base_path/1 目录下所有 .mat 文件的前缀作为被试编号
            - list[int]：指定要加载的被试编号，如 [1, 2, 3]
        :return: (features, labels, subjects)
            features: list，所有 trial 的特征（按被试顺序拼接）
            labels: ndarray，所有 trial 的标签
            subjects: ndarray，与 labels 等长，每个 trial 对应的 subject_id（用于 LOSO / Group split）
        """
        # 自动检测被试编号：以 session1 文件夹为准
        if subject_ids is None:
            session1_folder = os.path.join(base_path, "1")
            if not os.path.isdir(session1_folder):
                raise FileNotFoundError(f"Session 1 文件夹不存在: {session1_folder}")

            subject_ids = sorted(list({
                int(fname.split("_")[0])
                for fname in os.listdir(session1_folder)
                if fname.endswith(".mat") and "_" in fname and fname.split("_")[0].isdigit()
            }))

            if len(subject_ids) == 0:
                raise FileNotFoundError(f"在 {session1_folder} 未扫描到任何形如 '<subject>_*.mat' 的文件")

        all_features = []
        all_labels = []
        all_subjects = []

        for sid in subject_ids:
            print(f"\n👤 加载被试 {sid}")
            feat, lab = self.load_single_subject(sid, base_path)

            all_features.extend(feat)
            all_labels.append(lab)
            all_subjects.extend([sid] * len(lab))

        return all_features, np.concatenate(all_labels), np.array(all_subjects)

# ==========================================
# 示例运行逻辑（读取全部被试）
# ==========================================
if __name__ == "__main__":
    loader = SEEDIVLoader()

    # 1) 指向 eeg_feature_smooth 根目录（里面有 1/2/3 三个 session 文件夹）
    base_path = r"C:\Users\xinji\Desktop\archive\seed_iv\eeg_feature_smooth"

    try:
        # 2) 加载全部被试
        features, labels, subjects = loader.load_all_subjects(base_path=base_path)

        # 3) 输出检查
        print("\n" + "=" * 40)
        print("✅ 加载成功（全部被试）！")
        print(f"Total Trials: {len(features)}")
        print(f"Labels Shape: {labels.shape}")
        print(f"Subjects Shape: {subjects.shape}")
        print(f"Unique Subjects: {len(np.unique(subjects))}")

        # 检查第一个 trial 的维度
        first_trial_shape = features[0].shape
        print(f"First Trial Shape (channels, time, bands): {first_trial_shape}")

        # 打印前 10 条样本的 (subject, label)
        print("\n前 10 个 Trial 的 (Subject, Label -> Emotion):")
        for i in range(min(10, len(labels))):
            lab = int(labels[i])
            sid = int(subjects[i])
            print(f"Idx {i:04d}: Subject {sid:02d} | Label {lab} -> {loader.EMOTION_MAP[lab]}")

        # 打印各类分布
        counts = np.bincount(labels.astype(int), minlength=4)
        print("\n标签分布（0 neutral,1 sad,2 fear,3 happy）:")
        print(counts)

        print("=" * 40)

    except Exception as e:
        print(f"❌ 出错了: {e}")
