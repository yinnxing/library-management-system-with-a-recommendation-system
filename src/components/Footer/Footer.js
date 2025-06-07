import React from 'react';
import styles from './Footer.module.css';

const Footer = () => {
    const currentYear = new Date().getFullYear();

    const policies = [
        {
            icon: '📚',
            title: 'Hạn mức mượn tối đa',
            value: '5 cuốn/user',
            description: 'Số lượng sách đang mượn tối đa cùng lúc'
        },
        {
            icon: '⏰',
            title: 'Thời gian mượn tối đa',
            value: '14 ngày',
            description: 'Từ ngày mượn đến ngày trả dự kiến'
        },
        {
            icon: '💰',
            title: 'Phí quá hạn',
            value: '5.000₫/ngày',
            description: 'Tính từ ngày sau due_date'
        },
        {
            icon: '🔒',
            title: 'Thời gian giữ chỗ',
            value: '3 ngày',
            description: 'Từ lúc thông báo sách sẵn sàng đến khi user phải đến nhận'
        },
        {
            icon: '📋',
            title: 'Hạn mức đặt trước',
            value: '3 cuốn/user',
            description: 'Số lượng sách được giữ chỗ chờ mượn cùng lúc'
        }
    ];

    const quickLinks = [
        { name: 'Trang chủ', href: '/' },
        { name: 'Danh mục sách', href: '/books' },
        { name: 'Tìm kiếm', href: '/search' },
        { name: 'Tài khoản', href: '/profile' },
        { name: 'Lịch sử mượn', href: '/history' },
        { name: 'Hỗ trợ', href: '/support' }
    ];

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

                    {/* Quick Links Section */}
                    <div className={styles.section}>
                        <h4 className={styles.sectionTitle}>Liên kết nhanh</h4>
                        <ul className={styles.linkList}>
                            {quickLinks.map((link, index) => (
                                <li key={index}>
                                    <a href={link.href} className={styles.footerLink}>
                                        {link.name}
                                    </a>
                                </li>
                            ))}
                        </ul>
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

                    {/* Newsletter Section */}
                    <div className={styles.section}>
                        <h4 className={styles.sectionTitle}>Đăng ký nhận tin</h4>
                        <p className={styles.newsletterDescription}>
                            Nhận thông báo về sách mới và các hoạt động của thư viện
                        </p>
                        <div className={styles.newsletterForm}>
                            <input 
                                type="email" 
                                placeholder="Nhập email của bạn"
                                className={styles.newsletterInput}
                            />
                            <button className={styles.newsletterButton}>
                                Đăng ký
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Policies Section */}
            <div className={styles.policiesSection}>
                <div className={styles.container}>
                    <h3 className={styles.policiesTitle}>
                        <span className={styles.policiesIcon}>📋</span>
                        Chính sách chung
                    </h3>
                    <div className={styles.policiesGrid}>
                        {policies.map((policy, index) => (
                            <div key={index} className={styles.policyCard}>
                                <div className={styles.policyHeader}>
                                    <span className={styles.policyIcon}>{policy.icon}</span>
                                    <div className={styles.policyInfo}>
                                        <h5 className={styles.policyTitle}>{policy.title}</h5>
                                        <span className={styles.policyValue}>{policy.value}</span>
                                    </div>
                                </div>
                                <p className={styles.policyDescription}>{policy.description}</p>
                            </div>
                        ))}
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
                            <a href="/privacy" className={styles.legalLink}>Chính sách bảo mật</a>
                            <a href="/terms" className={styles.legalLink}>Điều khoản sử dụng</a>
                            <a href="/cookies" className={styles.legalLink}>Chính sách Cookie</a>
                        </div>
                    </div>
                </div>
            </div>
        </footer>
    );
};

export default Footer; 