import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import styles from './Signup.module.css';
import UserApi from '../../api/UserApi';
import { useUser } from '../../contexts/UserContext';

const Signup = () => {
    const { login } = useUser();
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        // Basic validation
        if (!username || !email || !password || !confirmPassword) {
            setError('Vui lòng điền đầy đủ tất cả các trường.');
            return;
        }

        if (password !== confirmPassword) {
            setError('Mật khẩu xác nhận không khớp.');
            return;
        }

        if (password.length < 6) {
            setError('Mật khẩu phải có ít nhất 6 ký tự.');
            return;
        }

        setIsLoading(true);

        try {
            const response = await UserApi.register(username, email, password);
            
            // Check if registration was successful (code: 0 indicates success)
            if (response.data.code === 0) {
                // Auto login after successful registration
                const loginResponse = await UserApi.login(email, password);
                
                if (loginResponse.data.code === 0) {
                    const loginData = loginResponse.data.result;
                    localStorage.setItem('accessToken', loginData.accessToken);
                    login({
                        userId: loginData.userId,
                        email: email,
                        role: loginData.role || 'USER'
                    }, loginData.accessToken);
                    
                    navigate('/');
                } else {
                    setError('Đăng nhập sau khi đăng ký thất bại. Vui lòng thử đăng nhập thủ công.');
                }
            } else {
                setError(response.data.message || 'Đăng ký thất bại.');
            }
        } catch (error) {
            setError('Lỗi kết nối đến máy chủ. Vui lòng thử lại sau.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className={styles.signupPage}>
            <div className={styles.signupContainer}>
                <div className={styles.signupHeader}>
                    <div className={styles.logoContainer}>
                        <span className={styles.logoIcon}>📚</span>
                        <h1 className={styles.logoText}>Harmony Library</h1>
                    </div>
                    <h2 className={styles.signupTitle}>Tạo Tài Khoản</h2>
                    <p className={styles.signupSubtitle}>
                        Tham gia cộng đồng độc giả và khám phá thế giới tri thức!
                    </p>
                </div>

                <div className={styles.signupForm}>
                    {error && (
                        <div className={styles.errorMessage}>
                            <span className={styles.errorIcon}>⚠️</span>
                            {error}
                        </div>
                    )}

                    <form onSubmit={handleSubmit}>
                        <div className={styles.formGroup}>
                            <label htmlFor="name" className={styles.formLabel}>
                                Họ và tên
                            </label>
                            <div className={styles.inputContainer}>
                                <span className={styles.inputIcon}>👤</span>
                                <input
                                    type="text"
                                    id="name"
                                    value={username}
                                    onChange={(e) => setUsername(e.target.value)}
                                    placeholder="Nhập họ và tên của bạn"
                                    className={styles.formInput}
                                    disabled={isLoading}
                                />
                            </div>
                        </div>

                        <div className={styles.formGroup}>
                            <label htmlFor="email" className={styles.formLabel}>
                                Email
                            </label>
                            <div className={styles.inputContainer}>
                                <span className={styles.inputIcon}>✉️</span>
                                <input
                                    type="email"
                                    id="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="Nhập địa chỉ email của bạn"
                                    className={styles.formInput}
                                    disabled={isLoading}
                                />
                            </div>
                        </div>

                        <div className={styles.formGroup}>
                            <label htmlFor="password" className={styles.formLabel}>
                                Mật khẩu
                            </label>
                            <div className={styles.inputContainer}>
                                <span className={styles.inputIcon}>🔒</span>
                                <input
                                    type="password"
                                    id="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="Nhập mật khẩu (ít nhất 6 ký tự)"
                                    className={styles.formInput}
                                    disabled={isLoading}
                                />
                            </div>
                        </div>

                        <div className={styles.formGroup}>
                            <label htmlFor="confirmPassword" className={styles.formLabel}>
                                Xác nhận mật khẩu
                            </label>
                            <div className={styles.inputContainer}>
                                <span className={styles.inputIcon}>🔐</span>
                                <input
                                    type="password"
                                    id="confirmPassword"
                                    value={confirmPassword}
                                    onChange={(e) => setConfirmPassword(e.target.value)}
                                    placeholder="Nhập lại mật khẩu của bạn"
                                    className={styles.formInput}
                                    disabled={isLoading}
                                />
                            </div>
                        </div>

                        <button 
                            type="submit" 
                            className={styles.signupButton}
                            disabled={isLoading}
                        >
                            {isLoading ? (
                                <>
                                    <div className={styles.spinner}></div>
                                    Đang tạo tài khoản...
                                </>
                            ) : (
                                <>
                                    <span className={styles.buttonIcon}>✨</span>
                                    Tạo Tài Khoản
                                </>
                            )}
                        </button>
                    </form>

                    <div className={styles.loginPrompt}>
                        <p>
                            Đã có tài khoản? 
                            <Link to="/login" className={styles.loginLink}>
                                Đăng nhập ngay
                            </Link>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Signup; 