import { useParams, useNavigate } from 'react-router-dom';
import books from '../../assets/books';
import styles from './BorrowBookPage.module.css'; 
import React, { useState, useEffect } from 'react';
import { useUser } from '../../contexts/UserContext';
import UserApi from '../../api/UserApi';

const BorrowBookPage = () => {
    const { bookId } = useParams();
    const { user } = useUser();
    const navigate = useNavigate();
    const [book, setBook] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [borrowStatus, setBorrowStatus] = useState({});
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Library policies data
    const libraryPolicies = [
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
        }
    ];

    const libraryInfo = {
        name: 'Harmony Library',
        address: '123 Đường Sách, Quận 1, TP.HCM',
        phone: '(028) 1234-5678',
        email: 'info@harmonylibrary.vn',
        hours: 'T2-T7: 8:00-20:00, CN: 9:00-17:00'
    };

    useEffect(() => {
        // Scroll to top when component loads
        window.scrollTo({ top: 0, behavior: 'smooth' });
        
        const fetchBook = async () => {
            try {
                setLoading(true);
                const response = await UserApi.getBook(bookId);
                setBook(response.data.result); 
                setLoading(false);
            } catch (err) {
                console.error("Error fetching book:", err);
                setError("Failed to load book information. Please try again later.");
                setLoading(false);
            }
        };

        fetchBook();
    }, [bookId]);

   const handleBorrowBook = async () => {
        if (!user) {
            setBorrowStatus({
                success: false,
                message: "Vui lòng đăng nhập để mượn sách.",
            });
            return;
        }

        try {
            setIsSubmitting(true);
        console.log(user.userId);
        const response = await UserApi.borrow(user.userId, parseInt(bookId));
        console.log(response);

            const { 
                transactionId, 
                bookTitle, 
                borrowDate, 
                pickupDeadline, 
                status, 
                message 
            } = response.data.result;
            
            console.log('Transaction ID:', transactionId);
            console.log('Pickup Deadline:', pickupDeadline);

        setBorrowStatus({
            success: true,
                message: message || "Yêu cầu mượn sách trực tuyến thành công!",
                transactionId: transactionId,
                bookTitle: bookTitle,
                borrowDate: borrowDate,
                pickupDeadline: pickupDeadline,
                status: status,
                libraryAddress: libraryInfo.address,
                contactInfo: `${libraryInfo.phone} / ${libraryInfo.email}`,
        });
    } catch (error) {
        console.error("Error borrowing book:", error);
            
            let errorMessage = "Không thể mượn sách. Vui lòng thử lại sau.";
            
            // Check if it's an API error response
            if (error.response?.data) {
                const { code, message } = error.response.data;
                
                // Handle specific error codes
                switch (code) {
                    case 1001:
                        errorMessage = "Bạn đã mượn quá số lượng sách cho phép.";
                        break;
                    case 1002:
                        errorMessage = "Sách này hiện không có sẵn để mượn.";
                        break;
                    case 1003:
                        errorMessage = "Bạn đã có yêu cầu mượn sách này đang chờ xử lý.";
                        break;
                    case 1004:
                        errorMessage = "Tài khoản của bạn đã bị khóa hoặc không có quyền mượn sách.";
                        break;
                    case 1005:
                        errorMessage = "Bạn có sách quá hạn chưa trả. Vui lòng trả sách trước khi mượn tiếp.";
                        break;
                    default:
                        // Use the message from API if available
                        errorMessage = message || errorMessage;
                        break;
                }
            } else if (error.request) {
                // Network error
                errorMessage = "Không thể kết nối đến máy chủ. Vui lòng kiểm tra kết nối mạng.";
            } else if (error.message) {
                // Other errors
                errorMessage = `Lỗi: ${error.message}`;
            }
            
        setBorrowStatus({
            success: false,
                message: errorMessage,
        });
        } finally {
            setIsSubmitting(false);
    }
};

    const handleBackToBooks = () => {
        navigate('/books');
    };

    if (loading) {
        return (
            <div className={styles.container}>
                <div className={styles.loadingSection}>
                    <div className={styles.loadingSpinner}></div>
                    <p className={styles.loadingText}>Đang tải thông tin sách...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className={styles.container}>
                <div className={styles.errorSection}>
                    <div className={styles.errorIcon}>⚠️</div>
                    <p className={styles.errorText}>{error}</p>
                    <button onClick={handleBackToBooks} className={styles.backButton}>
                        Quay lại danh sách sách
                    </button>
                </div>
            </div>
        );
    }

    if (!book) {
        return (
            <div className={styles.container}>
                <div className={styles.errorSection}>
                    <div className={styles.errorIcon}>📚</div>
                    <p className={styles.errorText}>Không tìm thấy sách.</p>
                    <button onClick={handleBackToBooks} className={styles.backButton}>
                        Quay lại danh sách sách
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className={styles.container}>
            {/* Header Section */}
            <div className={styles.headerSection}>
                <button onClick={handleBackToBooks} className={styles.backButton}>
                    ← Quay lại
                </button>
                <h1 className={styles.pageTitle}>
                    <span className={styles.titleIcon}>📖</span>
                    Mượn sách
                </h1>
            </div>

            {/* Main Content */}
            <div className={styles.mainContent}>
                {/* Book Information Card */}
                <div className={styles.bookCard}>
                    <div className={styles.bookHeader}>
                        <h2 className={styles.bookTitle}>{book.title}</h2>
                        <div className={styles.availabilityBadge}>
                            {book.availableQuantity > 0 ? (
                                <span className={styles.available}>✅ Có sẵn</span>
                            ) : (
                                <span className={styles.unavailable}>❌ Hết sách</span>
                            )}
                        </div>
                    </div>

                    <div className={styles.bookContent}>
                        <div className={styles.bookImageContainer}>
                <img
                    src={book.coverImageUrl}
                    alt={book.title}
                    className={styles.bookImage}
                />
                            {/* <div className={styles.imageOverlay}>
                                <span className={styles.overlayText}>Bìa sách chất lượng cao</span>
                            </div> */}
                        </div>

                <div className={styles.bookDetails}>
                            <div className={styles.detailItem}>
                                <span className={styles.detailIcon}>👤</span>
                                <div className={styles.detailContent}>
                                    <span className={styles.detailLabel}>Tác giả</span>
                                    <span className={styles.detailValue}>{book.author}</span>
                                </div>
                            </div>

                            <div className={styles.detailItem}>
                                <span className={styles.detailIcon}>📖</span>
                                <div className={styles.detailContent}>
                                    <span className={styles.detailLabel}>Nhà xuất bản</span>
                                    <span className={styles.detailValue}>{book.publisher}</span>
                                </div>
                            </div>

                            <div className={styles.detailItem}>
                                <span className={styles.detailIcon}>📅</span>
                                <div className={styles.detailContent}>
                                    <span className={styles.detailLabel}>Năm xuất bản</span>
                                    <span className={styles.detailValue}>{book.publicationYear}</span>
                                </div>
                            </div>

                            <div className={styles.detailItem}>
                                <span className={styles.detailIcon}>📚</span>
                                <div className={styles.detailContent}>
                                    <span className={styles.detailLabel}>Số lượng có sẵn</span>
                                    <span className={styles.detailValue}>{book.availableQuantity} cuốn</span>
                </div>
            </div>

            <button
                onClick={handleBorrowBook}
                                disabled={book.availableQuantity === 0 || isSubmitting}
                className={styles.borrowButton}
            >
                                {isSubmitting ? (
                                    <>
                                        <div className={styles.buttonSpinner}></div>
                                        Đang xử lý...
                                    </>
                                ) : (
                                    <>
                                        <span>📖</span>
                                        {book.availableQuantity > 0 ? 'Mượn sách' : 'Hết sách'}
                        </>
                    )}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Borrow Status */}
                {borrowStatus.message && (
                    <div className={`${styles.statusCard} ${borrowStatus.success ? styles.success : styles.error}`}>
                        <div className={styles.statusHeader}>
                            <span className={styles.statusIcon}>
                                {borrowStatus.success ? '✅' : '❌'}
                            </span>
                            <h3 className={styles.statusTitle}>
                                {borrowStatus.success ? 'Thành công!' : 'Có lỗi xảy ra'}
                            </h3>
                        </div>
                        
                        <p className={styles.statusMessage}>{borrowStatus.message}</p>
                        
                        {borrowStatus.success && borrowStatus.pickupDeadline && (
                            <div className={styles.borrowDetails}>
                                {borrowStatus.transactionId && (
                                    <div className={styles.borrowDetailItem}>
                                        <span className={styles.borrowDetailIcon}>🔖</span>
                                        <div>
                                            <span className={styles.borrowDetailLabel}>Mã giao dịch:</span>
                                            <span className={styles.borrowDetailValue}>{borrowStatus.transactionId}</span>
                                        </div>
                                    </div>
                                )}
                                
                                {borrowStatus.bookTitle && (
                                    <div className={styles.borrowDetailItem}>
                                        <span className={styles.borrowDetailIcon}>📚</span>
                                        <div>
                                            <span className={styles.borrowDetailLabel}>Tên sách:</span>
                                            <span className={styles.borrowDetailValue}>{borrowStatus.bookTitle}</span>
                                        </div>
                                    </div>
                                )}
                                
                                {borrowStatus.borrowDate && (
                                    <div className={styles.borrowDetailItem}>
                                        <span className={styles.borrowDetailIcon}>📅</span>
                                        <div>
                                            <span className={styles.borrowDetailLabel}>Ngày đặt mượn:</span>
                                            <span className={styles.borrowDetailValue}>
                                                {new Date(borrowStatus.borrowDate).toLocaleString('vi-VN')}
                                            </span>
                                        </div>
                                    </div>
                                )}
                                
                                <div className={styles.borrowDetailItem}>
                                    <span className={styles.borrowDetailIcon}>⏰</span>
                                    <div>
                                        <span className={styles.borrowDetailLabel}>Hạn nhận sách tại thư viện:</span>
                                        <span className={styles.borrowDetailValue}>
                                            {new Date(borrowStatus.pickupDeadline).toLocaleString('vi-VN')}
                                        </span>
                                    </div>
                                </div>
                                
                                {borrowStatus.status && (
                                    <div className={styles.borrowDetailItem}>
                                        <span className={styles.borrowDetailIcon}>📊</span>
                                        <div>
                                            <span className={styles.borrowDetailLabel}>Trạng thái:</span>
                                            <span className={styles.borrowDetailValue}>
                                                {borrowStatus.status === 'PENDING' ? 'Chờ nhận sách' : borrowStatus.status}
                                            </span>
                                        </div>
                                    </div>
                                )}
                                
                                <div className={styles.borrowDetailItem}>
                                    <span className={styles.borrowDetailIcon}>📍</span>
                                    <div>
                                        <span className={styles.borrowDetailLabel}>Địa chỉ thư viện:</span>
                                        <span className={styles.borrowDetailValue}>{borrowStatus.libraryAddress}</span>
                                    </div>
                                </div>
                                
                                <div className={styles.borrowDetailItem}>
                                    <span className={styles.borrowDetailIcon}>📞</span>
                                    <div>
                                        <span className={styles.borrowDetailLabel}>Thông tin liên hệ:</span>
                                        <span className={styles.borrowDetailValue}>{borrowStatus.contactInfo}</span>
                                    </div>
                                </div>
                                
                                <div className={styles.importantNote}>
                                    <span className={styles.noteIcon}>⚠️</span>
                                    <p>
                                        Vui lòng đến thư viện trong vòng 3 ngày để hoàn tất thủ tục mượn sách. 
                                        Nếu không, yêu cầu sẽ bị hủy tự động.
                                    </p>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Library Policies Section */}
                <div className={styles.policiesSection}>
                    <h3 className={styles.sectionTitle}>
                        <span className={styles.sectionIcon}>📋</span>
                        Chính sách mượn sách
                    </h3>
                    <div className={styles.policiesGrid}>
                        {libraryPolicies.map((policy, index) => (
                            <div key={index} className={styles.policyCard}>
                                <div className={styles.policyHeader}>
                                    <span className={styles.policyIcon}>{policy.icon}</span>
                                    <div className={styles.policyInfo}>
                                        <h4 className={styles.policyTitle}>{policy.title}</h4>
                                        <span className={styles.policyValue}>{policy.value}</span>
                                    </div>
                                </div>
                                <p className={styles.policyDescription}>{policy.description}</p>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Library Information Section */}
                <div className={styles.libraryInfoSection}>
                    <h3 className={styles.sectionTitle}>
                        <span className={styles.sectionIcon}>🏛️</span>
                        Thông tin thư viện
                    </h3>
                    <div className={styles.libraryInfoCard}>
                        <div className={styles.libraryHeader}>
                            <h4 className={styles.libraryName}>
                                <span className={styles.libraryIcon}>📖</span>
                                {libraryInfo.name}
                            </h4>
                        </div>
                        
                        <div className={styles.libraryDetails}>
                            <div className={styles.libraryDetailItem}>
                                <span className={styles.libraryDetailIcon}>📍</span>
                                <div>
                                    <span className={styles.libraryDetailLabel}>Địa chỉ:</span>
                                    <span className={styles.libraryDetailValue}>{libraryInfo.address}</span>
                                </div>
                            </div>
                            
                            <div className={styles.libraryDetailItem}>
                                <span className={styles.libraryDetailIcon}>📞</span>
                                <div>
                                    <span className={styles.libraryDetailLabel}>Điện thoại:</span>
                                    <span className={styles.libraryDetailValue}>{libraryInfo.phone}</span>
                                </div>
                            </div>
                            
                            <div className={styles.libraryDetailItem}>
                                <span className={styles.libraryDetailIcon}>✉️</span>
                                <div>
                                    <span className={styles.libraryDetailLabel}>Email:</span>
                                    <span className={styles.libraryDetailValue}>{libraryInfo.email}</span>
                                </div>
                            </div>
                            
                            <div className={styles.libraryDetailItem}>
                                <span className={styles.libraryDetailIcon}>🕒</span>
                                <div>
                                    <span className={styles.libraryDetailLabel}>Giờ mở cửa:</span>
                                    <span className={styles.libraryDetailValue}>{libraryInfo.hours}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BorrowBookPage;
