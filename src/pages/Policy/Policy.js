import React from 'react';
import styles from './Policy.module.css';

const Policy = () => {
    const policies = [
        {
            icon: '📚',
            title: 'Hạn mức mượn tối đa',
            value: '5 cuốn/người dùng',
            description: 'Mỗi người dùng có thể mượn tối đa 5 cuốn sách cùng một lúc'
        },
        {
            icon: '⏰',
            title: 'Thời gian mượn tối đa',
            value: '14 ngày',
            description: 'Thời gian mượn sách tối đa là 14 ngày kể từ ngày mượn'
        },
        {
            icon: '💰',
            title: 'Phí quá hạn',
            value: '5.000₫/ngày',
            description: 'Phí phạt cho mỗi ngày trả sách quá hạn'
        },
        {
            icon: '🔒',
            title: 'Thời gian giữ chỗ',
            value: '3 ngày',
            description: 'Thời gian giữ chỗ sách sau khi thông báo sẵn sàng'
        },
        {
            icon: '📋',
            title: 'Hạn mức đặt trước',
            value: '3 cuốn/người dùng',
            description: 'Số lượng sách có thể đặt trước cùng lúc'
        }
    ];

    const rules = [
        {
            title: 'Quy định về mượn sách',
            content: 'Người dùng phải đăng ký tài khoản và xác thực thông tin cá nhân trước khi có thể mượn sách. Mỗi lần mượn sách, người dùng cần xuất trình thẻ thư viện hoặc chứng minh nhân dân. Sách mượn phải được trả đúng thời hạn quy định và không được cho người khác mượn lại. Việc vi phạm các quy định này sẽ dẫn đến việc tạm ngừng quyền mượn sách.'
        },
        {
            title: 'Quy định về bảo quản sách',
            content: 'Người mượn có trách nhiệm giữ gìn sách sạch sẽ, không làm rách, bẩn hoặc viết vẽ lên sách. Việc photo copy toàn bộ nội dung sách là không được phép. Sách cần được bảo quản tránh ẩm ướt, côn trùng phá hoại và các tác động bên ngoài. Trong trường hợp làm mất hoặc hư hỏng sách, người mượn phải bồi thường theo quy định của thư viện.'
        },
        {
            title: 'Quy định về trật tự trong thư viện',
            content: 'Để tạo môi trường học tập tốt nhất, người dùng cần giữ im lặng và không gây tiếng ồn ảnh hưởng đến người khác. Việc ăn uống trong khu vực đọc sách là không được phép. Điện thoại cần được để ở chế độ im lặng hoặc rung khi ở trong thư viện. Mọi người dùng đều cần tuân thủ hướng dẫn và quy định của nhân viên thư viện.'
        }
    ];

    return (
        <div className={styles.policyPage}>
            <div className={styles.container}>
                {/* Header */}
                <div className={styles.header}>
                    <h1 className={styles.title}>
                        <span className={styles.titleIcon}>📋</span>
                        Chính sách & Quy định thư viện
                    </h1>
                    <p className={styles.subtitle}>
                        Các quy định và chính sách sử dụng dịch vụ thư viện Harmony Library
                    </p>
                </div>

                {/* Main Content Layout */}
                <div className={styles.mainContent}>
                    {/* Left Column */}
                    <div className={styles.leftColumn}>
                        {/* Policies Grid */}
                        <section className={styles.policiesSection}>
                            <h2 className={styles.sectionTitle}>Chính sách chung</h2>
                            <div className={styles.policiesGrid}>
                                {policies.map((policy, index) => (
                                    <div key={index} className={styles.policyCard}>
                                        <div className={styles.policyHeader}>
                                            <span className={styles.policyIcon}>{policy.icon}</span>
                                            <div className={styles.policyInfo}>
                                                <h3 className={styles.policyTitle}>{policy.title}</h3>
                                                <span className={styles.policyValue}>{policy.value}</span>
                                            </div>
                                        </div>
                                        <p className={styles.policyDescription}>{policy.description}</p>
                                    </div>
                                ))}
                            </div>
                        </section>

                        {/* Additional Info */}
                        <section className={styles.additionalInfo}>
                            <div className={styles.infoCard}>
                                <h3 className={styles.infoTitle}>
                                    <span className={styles.infoIcon}>ℹ️</span>
                                    Thông tin quan trọng
                                </h3>
                                <div className={styles.infoContent}>
                                    <p>
                                        Các chính sách và quy định này có thể được cập nhật theo thời gian. 
                                        Mọi thay đổi sẽ được thông báo trước ít nhất 7 ngày.
                                    </p>
                                    <p>
                                        Nếu có thắc mắc về các quy định, vui lòng liên hệ với nhân viên thư viện 
                                        hoặc gửi email tới <strong>info@harmonylibrary.vn</strong>
                                    </p>
                                </div>
                            </div>
                        </section>
                    </div>

                    {/* Right Column */}
                    <div className={styles.rightColumn}>
                        {/* Rules Section */}
                        <section className={styles.rulesSection}>
                            <h2 className={styles.sectionTitle}>Quy định chi tiết</h2>
                            <div className={styles.rulesContainer}>
                                {rules.map((rule, index) => (
                                    <div key={index} className={styles.ruleCard}>
                                        <h3 className={styles.ruleTitle}>{rule.title}</h3>
                                        <p className={styles.ruleContent}>{rule.content}</p>
                                    </div>
                                ))}
                            </div>
                        </section>

                        {/* Contact Section */}
                        <section className={styles.contactSection}>
                            <div className={styles.contactCard}>
                                <h3 className={styles.contactTitle}>
                                    <span className={styles.contactIcon}>📞</span>
                                    Liên hệ hỗ trợ
                                </h3>
                                <div className={styles.contactInfo}>
                                    <div className={styles.contactItem}>
                                        <span className={styles.contactLabel}>Điện thoại:</span>
                                        <span className={styles.contactValue}>(028) 1234-5678</span>
                                    </div>
                                    <div className={styles.contactItem}>
                                        <span className={styles.contactLabel}>Email:</span>
                                        <span className={styles.contactValue}>info@harmonylibrary.vn</span>
                                    </div>
                                    <div className={styles.contactItem}>
                                        <span className={styles.contactLabel}>Giờ làm việc:</span>
                                        <span className={styles.contactValue}>T2-T7: 8:00-20:00, CN: 9:00-17:00</span>
                                    </div>
                                </div>
                            </div>
                        </section>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Policy; 