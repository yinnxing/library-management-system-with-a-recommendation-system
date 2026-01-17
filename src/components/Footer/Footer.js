import React from 'react';
import styles from './Footer.module.css';

const Footer = () => {
    const currentYear = new Date().getFullYear();

    const contactInfo = [
        { icon: '📍', label: 'Địa chỉ', value: '123 Đường Sách, Quận 1, TP.HCM' },
        { icon: '📞', label: 'Điện thoại', value: '(028) 1234-5678' },
        { icon: '✉️', label: 'Email', value: 'info@harmonylibrary.vn' },
        { icon: '🕒', label: 'Giờ mở cửa', value: 'T2-T7: 8:00-20:00, CN: 9:00-17:00' }
    ];

    return (
        <footer className={styles.footer}>
            {/* Main Footer Content */}
            <div className={styles.footerContent}>
                <div className={styles.container}>
                    {/* Library Info Section */}
                    <div className={styles.section}>
                        <div className={styles.brandSection}>
                            <h3 className={styles.brandTitle}>
                                <span className={styles.brandIcon}>📖</span>
                                Harmony Library
                            </h3>
                            <p className={styles.brandDescription}>
                                Thư viện hiện đại với hệ thống quản lý sách tiên tiến, 
                                mang đến trải nghiệm đọc sách tuyệt vời cho mọi người.
                            </p>
                            <div className={styles.socialLinks}>
                                <a href="#" className={styles.socialLink} aria-label="Facebook">
                                    <span>📘</span>
                                </a>
                                <a href="#" className={styles.socialLink} aria-label="Twitter">
                                    <span>🐦</span>
                                </a>
                                <a href="#" className={styles.socialLink} aria-label="Instagram">
                                    <span>📷</span>
                                </a>
                                <a href="#" className={styles.socialLink} aria-label="YouTube">
                                    <span>📺</span>
                                </a>
                            </div>
                        </div>
                    </div>

                    {/* Contact Info Section */}
                    <div className={styles.section}>
                        <h4 className={styles.sectionTitle}>Thông tin liên hệ</h4>
                        <div className={styles.contactList}>
                            {contactInfo.map((contact, index) => (
                                <div key={index} className={styles.contactItem}>
                                    <span className={styles.contactIcon}>{contact.icon}</span>
                                    <div className={styles.contactContent}>
                                        <span className={styles.contactLabel}>{contact.label}:</span>
                                        <span className={styles.contactValue}>{contact.value}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Policy Section */}
                    <div className={styles.section}>
                        <h4 className={styles.sectionTitle}>Chính sách & Quy định</h4>
                        <p className={styles.policyDescription}>
                            Tìm hiểu về các quy định, chính sách và điều khoản sử dụng của thư viện
                        </p>
                        <div className={styles.policyLinks}>
                            <a href="/policy" className={styles.policyLink}>
                                <span className={styles.policyIcon}>📋</span>
                                Chính sách thư viện
                            </a>
                        </div>
                    </div>
                </div>
            </div>

            {/* Bottom Bar */}
            <div className={styles.bottomBar}>
                <div className={styles.container}>
                    <div className={styles.bottomContent}>
                        <div className={styles.copyright}>
                            <p>&copy; {currentYear} Harmony Library. Tất cả quyền được bảo lưu.</p>
                        </div>
                        <div className={styles.legalLinks}>
                            <a href="/policy" className={styles.legalLink}>Chính sách thư viện</a>
                            <a href="/privacy" className={styles.legalLink}>Chính sách bảo mật</a>
                            <a href="/terms" className={styles.legalLink}>Điều khoản sử dụng</a>
                        </div>
                    </div>
                </div>
            </div>
        </footer>
    );
};

export default Footer; 