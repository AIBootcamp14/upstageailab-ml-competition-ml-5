# write_log.py

import sys
from datetime import datetime

class Logger:
    """
    로그를 파일에 저장하고, 표준 출력(stdout)과 표준 에러(stderr)를
    로그 파일로 리디렉션하는 기능이 추가된 Logger 클래스
    """
    def __init__(self, log_path: str, print_also: bool = True):
        self.log_path = log_path
        self.print_also = print_also
        # 원본 표준 출력을 저장해 둡니다.
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        # 로그 파일을 열고, 라인 버퍼링을 사용합니다.
        self.log_file = open(log_path, 'a', encoding='utf-8', buffering=1)

    def write(self, message: str, print_also: bool = True, print_error: bool = False):
        """
        로그 메시지를 파일에 기록하고,
        print_also=True일 경우 콘솔에도 출력합니다.
        """
        # 메시지 앞뒤 공백을 제거하고, 개행 문자가 없으면 추가합니다.
        message = message.strip()
        if not message:
            return
            
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"{timestamp} | {message}\n"
        
        self.log_file.write(line)
        
        if self.print_also and print_also:
            if print_error:
                # 재귀 호출을 피하기 위해 원본 표준 출력을 사용합니다.
                self.original_stdout.write(f"\033[91m{line}\033[0m")
            else:
                self.original_stdout.write(line)
    
    def flush(self):
        """
        스트림 인터페이스에 필요한 flush 메서드입니다.
        """
        self.log_file.flush()

    def start_redirect(self):
        """
        표준 출력(stdout)과 표준 에러(stderr)를 이 로거 인스턴스로 리디렉션합니다.
        """
        self.write(">> 표준 출력 및 오류를 로그 파일로 리디렉션 시작", print_also=True)
        sys.stdout = self
        sys.stderr = self

    def stop_redirect(self):
        """
        표준 출력(stdout)과 표준 에러(stderr)를 원상 복구합니다.
        """
        self.write(">> 표준 출력 및 오류 리디렉션 중지", print_also=True)
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def close(self):
        """
        로그 파일을 닫습니다.
        """
        self.log_file.close()